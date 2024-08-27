# Materialize

The `materialize` transform writes out documents up to that point, marks the materialized path
as successful if execution is successful, and allows for reading from the materialized data as
a source. This transform is helpful if you are using show and take() as part of a notebook to
incrementally inspect output. You can use `materialize` to avoid re-computation.

Options:
* path: a Path or string represents the "directory" for the materialized elements. The filesystem
  and naming convention will be inferred.  The dictionary allowes finer control, and supports
  { root=Path|str, fs=pyarrow.fs, name=lambda Document -> str, clean=True,
    tobin=Document.serialize()}
  root is required

* source_mode: how this materialize step should be used as an input:
    * OFF: (default) does not act as a source

    * IF_PRESENT: If the materialize has successfully run to completion, or if the
        materialize step is the first step, use the contents of the directory as the
        inputs. WARNING: If you change the input files or any of the steps before the
        materialize step, you need to delete the materialize directory to force re-execution.
        
```python
## use materialize to write out intermediate and final state of a pipeline

import sycamore
docs = (
   sycamore.init()
   .read.binary(paths, binary_format="pdf")
   .partition(partitioner=SycamorePartitioner())
   # write results post partitioning
   .materialize(path="/tmp/partitioned")
   .regex_replace(COALESCE_WHITESPACE)
   .extract_entity(entity_extractor=OpenAIEntityExtractor(
       "title", llm=davinci_llm, prompt_template=title_template))
   # write just the titles out
   .materialize(path={root="/tmp/titles", tobin=lambda d: d.properties["title"].encode("utf-8"))
   .merge(merger=MarkedMerger())
   .spread_properties(["path"])
   .split_elements(tokenizer=tokenizer, max_tokens=512)
   .explode()
   .embed(embedder=SentenceTransformerEmbedder())
   # store all the data in S3 for sharing
   .materialize(path="s3://example-bucket/embedded-data")
   .take_all()
)

## use materialize as a data source, potentially from a different developer 
docs2 = sycamore.read.materialize(path="s3://example-bucket/embedded-data").take_all()
# docs and docs2 will be the same except for order

## use materialize as an intermediate cache
import sycamore
docs = (
   sycamore.init()
   .read.binary(paths, binary_format="pdf")
   .partition(partitioner=SycamorePartitioner())
   # write results post partitioning; on a second execution of the same or
   # a related pipeline, partitioning will not be repeated saving the time
   # for doing that work.
   .materialize(path="/tmp/partitioned", source_mode=sycamore.MaterializeSourceMode.IF_PRESENT)
   ...
)
```

