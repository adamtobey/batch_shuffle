import re

from .redis import rd, rd_user_key
from .config import Config
from .flush import trigger_flush

def _minify_json(blob):
    return re.sub('\s', '', blob)

def write_json(json_blob, name):
    append_line = _minify_json(json_blob)
    redis_key = rd_user_key(name, Config.redis_keys.wal)

    log_size = rd.lpush(redis_key, json_blob)

    # Queue a flush for every batch boundary. Since the flush process might
    # lag behind the ingestion, it's possible to reach a WAL with multiple
    # complete batches.
    if log_size % Config.batches.size == 0:
        trigger_flush(name)
