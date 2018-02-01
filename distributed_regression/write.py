from pdb import set_trace
import os
import re
import json
import pyarrow as pa
import numpy as np
from random import randrange, shuffle, sample

BASE_DIR = os.path.abspath('./data') # TODO configuration & proper path handling
WRITEAHEAD_DIR = 'writeahead'
BATCH_DIR = 'batches'

WAL_FILENAME = 'log'

BATCH_SIZE = 480 # TODO has to be divisible by n_batches factorial

# TODO logging

# TODO new algorithm:
# Process the WAL in chunks with same length as batches,
# init a constant size np array during decoding and then
# use that array to write.

# Blocks will be stored as serialized np arrays of fixed size
# to allow memory mapping

# To write, choose N random blocks and deserealize. Shuffle data
# between the blocks as follows:
# 1. Shuffle the new block since it's in order. The other blocks
# will still be shuffled from the end of this algorithm
# 2. Perform the memory map square dance
# 3. Shuffle each block
# 4. Serialize blocks (old ones overwrite)

# This algorithm (probabilistcally) ensures the following:
# * Each block has data from at least 5 non-contiguous sources
# * The randomness of blocks increases over time (last shuffle is important
# or else the square dance would just keep moving the same sub-blocks around)
# * There is negligable correlation between block age and data age

# TODO this is a dumb function
def data_path(*ext):
    return os.path.join(BASE_DIR, *ext)

def collapse_lines(blob):
    return re.sub('\s', '', blob) #TODO any reason to preserve whitespace?

def write_json(json_blob, name):
    path = data_path(name, WRITEAHEAD_DIR)
    append_line = collapse_lines(json_blob)

    # TODO handle file sharding
    with open(os.path.join(path, WAL_FILENAME), 'a') as out:
        out.write(append_line + '\n')

# TODO batch the wal on data collection; use redis to keep the count
# ensure it's thread-safe since requests may be handled by different
# servers simultaneously. Also ensure that data collection can continue
# with consistent state as the WAL is being ingested. The easiest way to
# do this would be to ingest only complete batches, which would also
# simplify a lot of the ensuing code. The only downside of this solution
# is that it could vastly increase the latency of the learning in cases
# where there is just not a lot of data.
def flush_to_batches(name):
    read_path = data_path(name, WRITEAHEAD_DIR)
    wal_path = os.path.join(read_path, WAL_FILENAME)

    processor = SchemaPreprocessor(name)

    batch = []
    with BatchWriter(name) as batch_writer:
        # TODO handle nonexistent directory
        with open(wal_path) as wal:
            count = 0
            for line in wal:
                batch.append(line)
                count += 1

                if count >= BATCH_SIZE:
                    # process a batch
                    batch_matrix = processor.json_blobs_to_matrix(batch)
                    batch_writer.write_batch_matrix(batch_matrix)

                    count = 0
                    batch = []

    # TODO inconsistent state if error before here
    with open(wal_path, 'w') as wal:
        # TODO uneccessary loops
        for line in batch:
            wal.write(line)

class SchemaViolationError(RuntimeError):
    pass

class SchemaPreprocessor(object):

    # TODO config
    SCHEMA_FILE = 'schema.json'

    def __init__(self, name):
        self.name = name

        self.init_from_directory(data_path(self.name))

    def init_from_directory(self, dir):
        schema_file = os.path.join(dir, self.SCHEMA_FILE)
        with open(schema_file) as schema:
            self.schema = json.load(schema)

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
    INIT_FILE = 'init.json'

    N_SHUFFLE_BATCHES = 4

    def __init__(self, name):
        self.name = name
        self.batch_dir = data_path(name, BATCH_DIR)

    # TODO handle missing directory
    def __enter__(self):
        init_file = os.path.join(data_path(self.name), self.INIT_FILE)
        with open(init_file) as init:
            init_state = json.load(init)

        self.n_batches = init_state['n_batches']
        return self

    # TODO handle write errors
    def __exit__(self, *args):
        init_file = os.path.join(data_path(self.name), self.INIT_FILE)
        init_state = dict(n_batches = self.n_batches)
        with open(init_file, 'w') as out:
            json.dump(init_state, out)

    def batch_file_name(self, batch_index):
        return os.path.join(self.batch_dir, str(batch_index))

    def choose_random_block(self):
        block_num = randrange(0, self.n_batches)
        return self.batch_file_name(block_num)

    def write_batch_matrix(self, batch_matrix):
        if self.n_batches >= self.N_SHUFFLE_BATCHES:
            self._write_batch_matrix_and_shuffle(batch_matrix)
        elif self.n_batches > 1:
            self._write_batch_matrix_and_shuffle(batch_matrix, self.n_batches)
        else:
            np.random.shuffle(batch_matrix)
            self.serialize_new_batch(batch_matrix)

    def sample_batches_without_replacement(self, sample_batches):
        batch_nums = sample(range(self.n_batches), k=sample_batches)
        return [self.batch_file_name(num) for num in batch_nums]

    def _write_batch_matrix_and_shuffle(self, batch_matrix, n_other_batches=N_SHUFFLE_BATCHES):
        # 1. Shuffle the incoming batch so that the data points it later
        # shares with other random batches are non-contiguous. The other
        # blocks are shuffled during this algorithm, so they don't need
        # to be shuffled here.
        np.random.shuffle(batch_matrix)

        # 2. Choose N_SHUFFLE_BATCHES other blocks at random to shuffle
        # with the new one
        shuffle_batch_index = {
            batch_path: self.deserialize_batch(batch_path)
            for batch_path in self.sample_batches_without_replacement(n_other_batches)
        }

        # 3. Interleave all the batches so that each has an equal amount of
        # data from each of the originals. Perform shuffle in place.
        # TODO what about the extra data that doesn't constitute a full block?
        n_batches = n_other_batches + 1

        assert BATCH_SIZE == batch_matrix.shape[0], "This algorithm currently only supports full batches"
        assert BATCH_SIZE % n_batches == 0, "This algorithm must be slightly modified to support batches that are not evenly divisible by self.N_SHUFFLE_BATCHES + 1 = {n_batches}."
        shuffle_block_size = batch_matrix.shape[0] // n_batches

        temp = np.ndarray((shuffle_block_size, batch_matrix.shape[1]))

        all_the_blocks = [batch_matrix, *shuffle_batch_index.values()]

        assert len(all_the_blocks) == n_batches, set_trace()

        # For each offset, each batch will get a block containing
        # BLOCK_SIZE / n_batches data points from the block `offset`
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
        # and they're deserialized into shared memory. They can be mutable
        # sometimes. Figure out the conditions to make it mutable. This
        # really should not be a problem. They're never truly shared here, so
        # they should be able to be mutable.
        with pa.memory_map(path) as batch_file:
            immutable_mofo = pa.deserialize(batch_file.read_buffer())
            nice_array = np.ndarray(immutable_mofo.shape)
            nice_array[:,:] = immutable_mofo[:,:]
            return nice_array

    def serialize_new_batch(self, matrix):
        path = self.batch_file_name(self.n_batches)

        # Must establish the memory mapped file before it can be opened as a
        # memory map
        with pa.OSFile(path, 'w') as new_file:
            new_file.write(pa.serialize(matrix).to_buffer())

        # TODO this counter is never updated from the actual files, so
        # it had better always be right. Handle IO errors properly and maybe
        # add some conditions when it should actually be measured as opposed to
        # continually updated.
        self.n_batches += 1

    def serialize_batch(self, matrix, path):
        with pa.memory_map(path, 'w') as batch_file:
            batch_file.write(pa.serialize(matrix).to_buffer())
