import json
import os
import pyarrow as pa
import numpy as np
from random import randrange, shuffle, sample
from .write import * # TODO refactor
from threading import Thread
import time

REDIS_FLUSH_KEY = 'flushes_needed'

def trigger_flush(name):
    rd.rpush(REDIS_FLUSH_KEY, name)

# TODO this implementation only allows a single flush worker.
# To scale, use the message channel only for triggering a flush,
# and put the flushed key in a queue. All available flush workers
# then race to read from that queue, but only one will be able to
# pop the key. If there are multiple keys, then they will execute
# concurrently this way.
class BatchFlushWorker(Thread):

    def __init__(self, redis_instance, done_signal):
        super().__init__()
        self.rd = redis_instance
        self.done_signal = done_signal
        self.total_time = 0

    def run(self):
        while True:
            response = self.rd.blpop(REDIS_FLUSH_KEY, 2)
            if response:
                name_to_flush = response[1].decode("utf-8")
                key_to_flush = wal_redis_key(name_to_flush)
                self.flush_to_batches(key_to_flush, name_to_flush)
            elif self.done_signal.is_set():
                print(f"Spent {self.total_time}s in flush")
                raise RuntimeError("I'm a savage...")

    def flush_to_batches(self, redis_key, name):
        start = time.perf_counter()
        processor = SchemaPreprocessor(name)
        batch_writer = BatchWriter(name)

        # Get the batch and remove the read range atomically
        with rd.pipeline() as pipe:
            pipe.multi()
            pipe.lrange(redis_key, 0, BATCH_SIZE - 1)
            pipe.ltrim(redis_key, BATCH_SIZE, -1)
            batch = pipe.execute()[0]

        batch_matrix = processor.json_blobs_to_matrix(batch)
        batch_writer.write_batch_matrix(batch_matrix)

        self.total_time += time.perf_counter() - start

class SchemaViolationError(RuntimeError):
    pass

class SchemaPreprocessor(object):

    # TODO config
    SCHEMA_KEY = 'schema'

    def __init__(self, name):
        self.name = name
        self.schema = json.loads(rd.get(rd_user_key(self.name, self.SCHEMA_KEY)).decode('utf-8'))
        # TODO this will change
        self.data_dimension = len(self.schema)

    def json_blobs_to_matrix(self, json_blobs):
        out = np.ndarray((BATCH_SIZE, self.data_dimension))
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

    # TODO config
    N_BATCH_KEY = 'n_batches'

    N_SHUFFLE_BATCHES = 4

    def __init__(self, name):
        self.name = name
        self.batch_dir = data_path(name, BATCH_DIR)
        self.n_batch_key = rd_user_key(self.name, self.N_BATCH_KEY)

    def n_batches(self):
        # TODO when does this first get set?
        return int(rd.get(self.n_batch_key).decode('utf-8'))

    def incr_n_batches(self):
        rd.incr(self.n_batch_key)

    def batch_file_name(self, batch_index):
        return os.path.join(self.batch_dir, str(batch_index))

    def choose_random_block(self):
        block_num = randrange(0, self.n_batches())
        return self.batch_file_name(block_num)

    def write_batch_matrix(self, batch_matrix):
        n_batches = self.n_batches()
        if n_batches >= self.N_SHUFFLE_BATCHES:
            self._write_batch_matrix_and_shuffle(batch_matrix)
        elif n_batches > 1:
            self._write_batch_matrix_and_shuffle(batch_matrix, n_batches)
        else:
            np.random.shuffle(batch_matrix)
            self.serialize_new_batch(batch_matrix)

    def sample_batches_without_replacement(self, sample_batches):
        batch_nums = sample(range(self.n_batches()), k=sample_batches)
        return [self.batch_file_name(num) for num in batch_nums]

    def _write_batch_matrix_and_shuffle(self, batch_matrix, n_other_batches=N_SHUFFLE_BATCHES):
        # 1. Shuffle the incoming batch so that the data points it later
        # shares with other random batches are non-contiguous. The other
        # blocks are shuffled during this algorithm, so they don't need
        # to be shuffled here.
        np.random.shuffle(batch_matrix)

        # 2. Choose n_shuffle_batches other blocks at random to shuffle
        # with the new one
        shuffle_batch_index = {
            batch_path: self.deserialize_batch(batch_path)
            for batch_path in self.sample_batches_without_replacement(n_other_batches)
        }

        # 3. Interleave all the batches so that each has an equal amount of
        # data from each of the originals. perform shuffle in place.
        # TODO what about the extra data that doesn't constitute a full block?
        n_batches = n_other_batches + 1

        assert BATCH_SIZE == batch_matrix.shape[0], "This algorithm currently only supports full batches"
        assert BATCH_SIZE % n_batches == 0, "This algorithm must be slightly modified to support batches that are not evenly divisible by self.n_shuffle_batches + 1 = {n_batches}."
        shuffle_block_size = batch_matrix.shape[0] // n_batches

        temp = np.ndarray((shuffle_block_size, batch_matrix.shape[1]))

        all_the_blocks = [batch_matrix, *shuffle_batch_index.values()]

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
        self.serialize_new_batch(batch_matrix)

        for path, batch in shuffle_batch_index.items():
            self.serialize_batch(batch, path)

    def deserialize_batch(self, path):
        # TODO this is fucking insane. NP arrays are stored as arrow tensors
        # and they're deserialized into shared memory. they can be mutable
        # sometimes. figure out the conditions to make it mutable. this
        # really should not be a problem. they're never truly shared here, so
        # they should be able to be mutable.
        with pa.memory_map(path) as batch_file:
            immutable_mofo = pa.deserialize(batch_file.read_buffer())
            nice_array = np.ndarray(immutable_mofo.shape)
            nice_array[:,:] = immutable_mofo[:,:]
            return nice_array

    def serialize_new_batch(self, matrix):
        path = self.batch_file_name(self.n_batches())

        # Must establish the memory mapped file before it can be opened as a
        # memory map
        with pa.OSFile(path, 'w') as new_file:
            new_file.write(pa.serialize(matrix).to_buffer())

        # TODO this counter is never updated from the actual files, so
        # It had better always be right. Handle io errors properly and maybe
        # add some conditions when it should actually be measured as opposed to
        # continually updated.
        self.incr_n_batches()

    def serialize_batch(self, matrix, path):
        with pa.memory_map(path, 'w') as batch_file:
            batch_file.write(pa.serialize(matrix).to_buffer())
