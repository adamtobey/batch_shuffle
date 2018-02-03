import os
from math import factorial
from collections import namedtuple

# TODO this seems like a hacky solution to a common problem.
# Look around for an existing, more familiar solution.
def _params(**params):
    P = namedtuple("ParamBundle", params.keys())
    return P(**params)

# TODO allow for overriding configs
class Config(object):

    redis = _params(
        host='localhost',
        port=6379,
        db=0
    )

    redis_keys = _params(
        flush_queue='flushes_needed',
        wal='WAL',
        schema='schema',
        n_batches='n_batches'
    )

    directories = _params(
        batches='batches'
    )

    batches = _params(
        size=480,
        shuffle_factor=4
    )

    data = _params(
        path=os.path.abspath('./data')
    )

# Verify configuration is valid
assert Config.batches.size % factorial(Config.batches.shuffle_factor) == 0, "The batch size currently must be divisible by the factorial of n_shuffle_batches so that it can evenly shuffle a batch between N other batches for N <= n_shuffle_batches."
