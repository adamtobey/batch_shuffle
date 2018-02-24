# About

This is a simple project I built to test out an idea I had for storing large online datasets such as continuously streaming analytics and usage data for a product. This project continuously takes in data and stores it on disk in pre-computed batches, ensuring the resulting batches don't have contiguous data and there is no significant correlation between batch age and data age. Simply writing data points to the newest batch violates the precondition of most batched machine learning algorithms and versions of stochastic gradient descent, that the batches represent a random sample from the data generating distribution.

# How it works
The data is kept in a writeahead log initially, allowing the write to have consistent and low latency, ideal for use in an API accepting realtime data. The log is flushed to disk whenever it represents a complete batch by background worker processes using Dask Distributed, suitable for multi-tenant applications. However, the flush code is not thread-safe so batches for a single dataset are managed with a distributed lock.

The flush step creates a new batch in memory then samples N (configurable) existing batches to shuffle with. The batches are shuffled together with low memory and CPU overhead using Numpy permutations and swapping contiguous memory blocks. Each batch will then contain equal amounts of data from each other batch. Because old batches are shuffled with each new batch, the data gets more dispersed over time, each batch approximating a random sample from the data distribution. See `BatchWriter#_write_batch_matrix_and_shuffle` in flush.py for implementation of this algorithm.

Blocks are represented as Numpy arrays and serialized using PyArrow. Existing blocks are opened and written using Arrow memory maps with very low overhead on garbage collection and deserialization of string-based formats.
