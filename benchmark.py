from importlib import import_module
import sys

# TODO fuck this
import_module(f"benchmarks.{sys.argv[1]}")
