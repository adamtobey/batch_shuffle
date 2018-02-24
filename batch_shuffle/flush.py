import json
import numpy as np
import time

from random import shuffle
from dask.distributed import Lock, Reschedule

from .cluster import client
from .redis import rd, rd_user_key
from .config import Config
from .batch_manager import BatchManager

def trigger_flush(name):
    future = client.submit(_flush_to_batches, rd_user_key(name, Config.redis_keys.wal), name)

def _flush_to_batches(redis_key, name):
    lock = Lock(redis_key)
    # TODO set timeout and handle
    if lock.acquire(timeout=1):
        try:
            processor = SchemaPreprocessor(name)
            batch_writer = BatchWriter(name)

            # Get the batch and remove the read range atomically
            with rd.pipeline() as pipe:
                pipe.multi()
                pipe.lrange(redis_key, 0, Config.batches.size - 1)
                pipe.ltrim(redis_key, Config.batches.size, -1)
                batch = pipe.execute()[0]

            batch_matrix = processor.json_blobs_to_matrix(batch)
            batch_writer.write_batch_matrix(batch_matrix)
        finally:
            lock.release()
    else:
        raise Reschedule()




class SchemaViolationError(RuntimeError):
    pass

class SchemaPreprocessor(object):

    def __init__(self, name):
        self.name = name
        self.schema = json.loads(rd.get(rd_user_key(self.name, Config.redis_keys.schema)).decode('utf-8'))
        # TODO this will change
        self.data_dimension = len(self.schema)

    def json_blobs_to_matrix(self, json_blobs):
        out = np.ndarray((Config.batches.size, self.data_dimension))
        for row, json_blob in enumerate(json_blobs):
            obj = json.loads(json_blob)
            for col, (name, type_name) in enumerate(self.schema.items()):
                if name not in obj:
                    raise SchemaViolationError(f"Required field {name} is missing from data")
                else:
                    out[row, col] = self.interpret_with_type(type_name, obj[name], name)
        return out

    # TODO handle more types
    def interpret_with_type(self, type_name, value, name):
        if type_name == "numeric":
            if isinstance(value, (int, float)):
                return value
            else:
                raise SchemaViolationError(f"Field {name} was declared as numeric but is {type(value)}")
        else:
            raise SchemaViolationError(f"Unknown type {type_name} for field {name}.")

class BatchWriter(object):

    def __init__(self, name):
        self.bm = BatchManager(name)

    def write_batch_matrix(self, batch_matrix):
        n_batches = self.bm.batch_count()
        if n_batches >= Config.batches.shuffle_factor:
            self._write_batch_matrix_and_shuffle(batch_matrix)
        elif n_batches > 1:
            self._write_batch_matrix_and_shuffle(batch_matrix, n_batches)
        else:
            np.random.shuffle(batch_matrix)
            self.bm.serialize_new_batch(batch_matrix)

    def _write_batch_matrix_and_shuffle(self, batch_matrix, n_other_batches=Config.batches.shuffle_factor):
        assert Config.batches.size == batch_matrix.shape[0], "This algorithm currently only supports full batches"

        # 1. Shuffle the incoming batch so that the data points it later
        # shares with other random batches are non-contiguous. The other
        # blocks are shuffled during this algorithm, so they don't need
        # to be shuffled here.
        np.random.shuffle(batch_matrix)

        # 2. Choose n_shuffle_batches other blocks at random to shuffle
        # with the new one
        shuffle_batch_index = {
            batch_name: self.bm.deserialize_batch(batch_name)
            for batch_name in self.bm.sample_batches_without_replacement(n_other_batches)
        }

        # 3. Interleave all the batches so that each has an equal amount of
        # data from each of the originals. perform shuffle in place.
        n_batches = n_other_batches + 1
        shuffle_block_size = batch_matrix.shape[0] // n_batches

        all_the_blocks = [batch_matrix, *shuffle_batch_index.values()]
        temp = np.ndarray((shuffle_block_size, batch_matrix.shape[1]))

        # For each offset, each batch will get a block containing
        # block_size / n_batches data points from the block `offset`
        # indices away. The cycle starts with the last recipient
        # putting its block into the temp storage and ends with that
        # block going to the last donor batch. The offsets start at
        # 1 since the zero offset case corresponds to no change.
        for offset in range(1, n_batches):
            block_slice = slice(
                offset * shuffle_block_size,
                (offset + 1) * shuffle_block_size
            )
            temp[:,:] = all_the_blocks[0][block_slice]
            into_matrix = all_the_blocks[0]
            for i in range(offset, n_batches, offset):
                from_matrix = all_the_blocks[i % n_batches]
                into_matrix[block_slice] = from_matrix[block_slice]
                into_matrix = from_matrix
            into_matrix[block_slice] = temp[:,:]

        # 5. Shuffle all the batches so that future interleaves (which share
        # the same indexes) will continue to randomly propagate the data
        # instead of simply swapping the same blocks between batches.
        np.random.shuffle(batch_matrix)
        for batch in shuffle_batch_index.values():
            np.random.shuffle(batch)

        # 6. Serialize the batches
        self.bm.serialize_new_batch(batch_matrix)

        for batch_name, batch_matrix in shuffle_batch_index.items():
            self.bm.serialize_batch(batch_matrix, batch_name)
