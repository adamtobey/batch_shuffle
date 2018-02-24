from dask.distributed import Client

from .config import Config

client = Client(Config.cluster.address)
