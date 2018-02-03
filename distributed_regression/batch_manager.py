import os
import pyarrow as pa
import numpy as np
import pathlib

from random import sample

from .redis import rd, rd_user_key
from .config import Config

class BatchManager(object):

    def __init__(self, name):
        self.name = name
        self.batch_dir = os.path.join(Config.data.path, self.name, Config.directories.batches)
        self.n_batch_key = rd_user_key(self.name, Config.redis_keys.n_batches)

        self._init_dir()

    # TODO if n_batches is set, assume nothing needs to be done to
    # reduce latency in creating a BatchManager
    def _init_dir(self):
        # Make data directory if one doesn't exist yet
        pathlib.Path(self.batch_dir).mkdir(parents=True, exist_ok=True)

        # Set n_batches if not currently set, counting the batches from
        # the filesystem if necessary
        current_n_batches = rd.get(self.n_batch_key)
        if current_n_batches == None:
            fs_n_batches = len(os.listdir(self.batch_dir))
            rd.set(self.n_batch_key, fs_n_batches)

    def batch_count(self):
        return int(rd.get(self.n_batch_key).decode('utf-8'))

    def _incr_batch_count(self):
        rd.incr(self.n_batch_key)

    def _batch_path(self, batch_index):
        return os.path.join(self.batch_dir, str(batch_index))

    def sample_batches_without_replacement(self, num_batches):
        batch_nums = sample(range(self.batch_count()), k=num_batches)
        return [str(num) for num in batch_nums]

    def deserialize_batch(self, batch):
        # TODO NP arrays are stored as arrow tensors
        # and they're deserialized into shared memory. they can be mutable
        # sometimes. figure out the conditions to make it mutable. this
        # really should not be a problem. they're never truly shared here, so
        # they should be able to be mutable.
        with pa.memory_map(self._batch_path(batch)) as batch_file:
            immutable_mofo = pa.deserialize(batch_file.read_buffer())
            nice_array = np.ndarray(immutable_mofo.shape)
            nice_array[:,:] = immutable_mofo[:,:]
            return nice_array

    def serialize_new_batch(self, matrix):
        path = self._batch_path(self.batch_count())

        # Must establish the memory mapped file before it can be opened as a
        # memory map
        with pa.OSFile(path, 'w') as new_file:
            new_file.write(pa.serialize(matrix).to_buffer())

        self._incr_batch_count()

    def serialize_batch(self, matrix, batch):
        with pa.memory_map(self._batch_path(batch), 'w') as batch_file:
            batch_file.write(pa.serialize(matrix).to_buffer())
