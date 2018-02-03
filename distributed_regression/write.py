import redis
import os
import re

BASE_DIR = os.path.abspath('./data') # TODO configuration & proper path handling
BATCH_DIR = 'batches'

BATCH_SIZE = 480 # TODO has to be divisible by n_batches factorial

rd = redis.StrictRedis(host="localhost", port=6379, db=0)

def rd_user_key(user_name, key):
    return f"{user_name}:{key}"

# TODO another dumb function
def wal_redis_key(name):
    return f"{name}:WAL"

# TODO logging

# TODO this is a dumb function
def data_path(*ext):
    return os.path.join(BASE_DIR, *ext)

def collapse_lines(blob):
    return re.sub('\s', '', blob) #TODO any reason to preserve whitespace?

from .flush import trigger_flush

def write_json(json_blob, name):
    append_line = collapse_lines(json_blob)
    redis_key = wal_redis_key(name)

    log_size = rd.lpush(redis_key, json_blob)

    # Queue a flush for every batch boundary. Since the flush process might
    # lag behind the ingestion, it's possible to reach a WAL with multiple
    # complete batches.
    if log_size % BATCH_SIZE == 0:
        trigger_flush(name)
