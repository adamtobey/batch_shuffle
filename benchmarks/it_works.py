import os
import pathlib
import shutil
import json
import numpy as np
import pyarrow as pa
from random import random
import time
from threading import Event

from distributed_regression.config import Config
from distributed_regression.flush import BatchFlushWorker
from distributed_regression.write import write_json
from distributed_regression.redis import rd, rd_user_key
from distributed_regression.batch_manager import BatchManager

TEST_DIR = "test"
DIR_PATH = os.path.join(Config.data.path, TEST_DIR)

N_BATCHES = 20
N_POINTS = N_BATCHES * Config.batches.size

def init():
    batch_path = os.path.join(DIR_PATH, Config.directories.batches)
    pathlib.Path(batch_path).mkdir(parents=True, exist_ok=True)

    schema = json.dumps(dict(
        x='numeric',
        y='numeric',
        seq='numeric'
    ))
    rd.set(rd_user_key(TEST_DIR, Config.redis_keys.schema), schema)

    rd.set(rd_user_key(TEST_DIR, Config.redis_keys.n_batches), 0)

def perform_test():
    total_time = 0
    for seq in range(N_POINTS):
        point_json = json.dumps(dict(
            x=random(),
            y=random(),
            seq=seq
        ))
        start = time.perf_counter()
        write_json(point_json, TEST_DIR)
        total_time += time.perf_counter() - start
    print(f"Wrote {N_POINTS} points in {total_time}s")

def test_data_corruption():
    seq_count = {
        seq: 0 for seq in range(N_POINTS)
    }
    batch_path = os.path.join(Config.data.path, TEST_DIR, Config.directories.batches)
    bm = BatchManager(TEST_DIR)
    for batch in os.listdir(batch_path):
        path = os.path.join(batch_path, batch)
        mat = bm.deserialize_batch(path)
        for seq in mat[:,2]:
            seq_count[seq] += 1

    uh_oh = False
    for seq, count in seq_count.items():
        if count != 1:
            uh_oh = True
            print(f"Seq {seq} was found {count} times!")
    if not uh_oh:
        print("Data is consistent.")

def output_distribution():
    batch_path = data_path(TEST_DIR, Config.directories.batches)
    bm = BatchManager(TEST_DIR)
    batches = sorted([int(name) for name in os.listdir(batch_path)])
    max_batch = max(batches)
    mean_seqs = np.ndarray((max_batch + 1,))
    for batch in batches:
        path = os.path.join(batch_path, str(batch))
        mat = bm.deserialize_batch(path)
        mean_seq = mat[:,2].mean()
        mean_seqs[batch] = mean_seq
    print("Mean seqs: ", mean_seqs)
    print("Correlation: ", np.corrcoef(batches, mean_seqs))

rd.flushall()
init()
done_signal = Event()
BatchFlushWorker(rd, done_signal).start()
perform_test()
time.sleep(2)
done_signal.set()
test_data_corruption()
# output_distribution()
