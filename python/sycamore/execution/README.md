# Execution

Execution is the layer taking care of execution plans, we inherit the similar
concept from database. The core of each operator is the `execute` method which
triggers the physical layer execution. The physical layer is currently built
on top of ray Dataset. Dataset itself has an Execution layer, it focuses more
on optimization like fusing pipelining tasks to avoid ray scheduling overhead.

This has advantages in couple dimensions:

1. Show a clear lineage of execution
2. Lazy execution gives opportunities for preprocessing execution plans, e.g.
   adjusting partition size, modifying ray remote parameters to make it easier
   to fuse.

Execution are basically categorized into scans, transforms and writes.

## Scans
Taking care of reading data from different data sources.

## Transforms

## Writes

## Kernels
Kernels are low-level primitives which are executed as task/actor in worker
nodes. We extract these blocks out from operators just for potential easier
reuse purpose.
