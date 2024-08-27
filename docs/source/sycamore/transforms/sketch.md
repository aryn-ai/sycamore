## Sketch
The `sketch` transform adds metadata to each Document containing a sketch that can be used to identify near-duplicate documents.  This process is the prerequisite for later removing or collapsing near-duplicate documents.  Currently, the sketch consists of a set of hash values called `shingles`.  These are relatively inexpensive to calculate and can safely be a default part of any ingestion pipeline.  Using `sketch` in a Sycamore data prep pipeline is relatively easy:

```python
docset = (context.read.binary(...)
          .partition(...)
	  .explode()
          .sketch()
	  .embed(...))
```

Query-time de-duplication is explained [here](../querying_data/dedup.md).  For more information, see the documentation for [Sketcher](../../APIs/transforms/sketcher.html#sycamore.transforms.sketcher.Sketcher).
