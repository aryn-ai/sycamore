## LLM Query
To be able to query the document elements, Sycamore provides a LLM Query transform that can be used to query the text in the document and its elements. To do so, it instantiates a LLMTextQueryAgent that can be used to execute queries on the document.

### LLMTextQueryAgent

The LLMTextQueryAgent allows users to execute queries on the document and its elements. The following parameters are supported:

Parameters:
* ```prompt```: A prompt to be passed into the underlying LLM execution engine
* ```llm```: The LLM Client to be used here. It is defined as an instance of the {LLM}`API documentation </APIs/data_preparation/docsetwriter>` class in Sycamore.
* ```output_property```: (optional, default=`"llm_response"`) The output property of the document or element to add results in.
* ```format_kwargs```: (optional, default=`None`) If passed in, details the formatting details that must be passed into the underlying Jinja Sandbox.
* ```per_element```: (optional, default=`True`) Whether to execute the call per each element or on the Document itself.
* ```number_of_elements```: (optional, default=`None`) When `per_element` is true, limits the number of elements to add an `output_property`. Otherwise, the response is added to the entire document using a limited prefix subset of the elements.
* ```llm_kwargs```:(optional, default=`{}`) Keyword arguments to be passed into the underlying LLM execution engine.
* `element_type`: (optional) Parameter to only execute the LLM query on a particular element type. If not specified, the query will be executed on all elements.

Here is an example of querying the elements of the document with the LLMTextQueryAgent:

```Python
ctx = sycamore.init()
docset = ctx.read.binary(s3://my-bucket/my-folder/, binary_format="pdf")
            .partition(partitioner=ArynPartitioner(extract_table_structure=True))
```
