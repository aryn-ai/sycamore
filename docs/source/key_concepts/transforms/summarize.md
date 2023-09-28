## Summarize
Similar to the extract entity transform, the summarize transform generates summaries of documents or elements. The ``LLMElementTextSummarizer`` summarizes a subset of the elements from each Document. It takes an LLM implementation and a callable specifying the subset of elements to summarize. The following examples shows how to use this transform to summarize elements that are longer than a certain length.

```python
def filter_elements_on_length(
    document: Document,
    minimum_length: int = 10,
) -> list[Element]:
    def filter_func(element: Element):
        if element.text_representation is not None:
            return len(element.text_representation) > minimum_length

    return filter_elements(document, filter_func)

llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)

docset = docset.summarize(LLMElementTextSummarizer(llm, filter_elements_on_length))
```
