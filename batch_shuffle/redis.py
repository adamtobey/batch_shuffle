import redis

from .config import Config

rd = redis.StrictRedis(host=Config.redis.host, port=Config.redis.port, db=Config.redis.db)

def rd_user_key(user_name, key):
    return f"{user_name}:{key}"
