import os
import pathlib
import shutil
import json
import numpy as np
import csv
import pyarrow as pa
from random import random
import time

from distributed_regression.write import write_json, BASE_DIR, WRITEAHEAD_DIR, BATCH_DIR, data_path, flush_to_batches, BATCH_SIZE

TEST_DIR = "test"
DIR_PATH = os.path.join(BASE_DIR, TEST_DIR)

N_BATCHES = 200
N_POINTS = N_BATCHES * BATCH_SIZE

def cleanup():
    shutil.rmtree(data_path(TEST_DIR, WRITEAHEAD_DIR))
    shutil.rmtree(data_path(TEST_DIR, BATCH_DIR))

def init():
    pathlib.Path(data_path(TEST_DIR, WRITEAHEAD_DIR)).mkdir(parents=True, exist_ok=True)

    batch_path = data_path(TEST_DIR, BATCH_DIR)
    pathlib.Path(batch_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(DIR_PATH, 'schema.json'), 'w') as out:
        json.dump(dict(
            x='numeric',
            y='numeric',
            seq='numeric'
        ), out)

    with open(os.path.join(DIR_PATH, 'init.json'), 'w') as out:
        json.dump(dict(
            n_batches=0
        ), out)

def perform_test():
    start_gen = time.perf_counter()
    for seq in range(N_POINTS):
        point_json = json.dumps(dict(
            x=random(),
            y=random(),
            seq=seq
        ))
        write_json(point_json, TEST_DIR)
    print(f"Generated {N_POINTS} points in {time.perf_counter() - start_gen}s")

    start_flush = time.perf_counter()
    flush_to_batches(TEST_DIR)
    print(f"Flushed {N_POINTS} points in {time.perf_counter() - start_flush}s")

def block_to_np(file):
    out = []
    reader = csv.reader(file)
    for row in reader:
        out.append(np.array([float(d) for d in row]))
    return np.array(out)

# TODO this copy paste should be resolved with refactor
def deserialize_batch(path):
    with pa.memory_map(path) as batch_file:
        immutable_mofo = pa.deserialize(batch_file.read_buffer())
        nice_array = np.ndarray(immutable_mofo.shape)
        nice_array[:,:] = immutable_mofo[:,:]
        return nice_array

def test_data_corruption():
    seq_count = {
        seq: 0 for seq in range(N_POINTS)
    }
    batch_path = data_path(TEST_DIR, BATCH_DIR)
    for batch in os.listdir(batch_path):
        path = os.path.join(batch_path, batch)
        mat = deserialize_batch(path)
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
    batch_path = data_path(TEST_DIR, BATCH_DIR)
    batches = sorted([int(name) for name in os.listdir(batch_path)])
    max_batch = max(batches)
    mean_seqs = np.ndarray((max_batch + 1,))
    for batch in batches:
        path = os.path.join(batch_path, str(batch))
        mat = deserialize_batch(path)
        mean_seq = mat[:,2].mean()
        mean_seqs[batch] = mean_seq
    print("Mean seqs: ", mean_seqs)
    print("Correlation: ", np.corrcoef(batches, mean_seqs))

# cleanup()
init()
perform_test()
# test_data_corruption()
output_distribution()
