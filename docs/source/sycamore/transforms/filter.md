## Filter

The filter transform lets you retain or discard documents from a DocSet based on a predicate. For example, the following code filters a DocSet to retain only those Documents containing at least 2000 characters.

```python
docset.filter(lambda doc: sum(len(el.text_representation)
                              for el in doc.elements
                              if el.text_representation is not None) >= 2000)
```

We used a lambda function here, which works well for simple funtions, but we can also use any Python callable. For instance, if we frequently want to filter by different document lengths, we can create a `LengthFilter` class, and pass that to the `filter` function.

```python
class LengthFilter:
    def __init__(self, length: int):
        self.length = length

    def __call__(self, doc: Document) -> Document:
        total = 0
        for el in doc.elements:
            if el.text_representation is not None:
                total += len(el.text_representation)
        return total >= self.length

docset.filter(LengthFilter(2000))
```

Note that `__call__` must take a single argument of type `Document` and return a `Document`, but the `__init__` method can take additional parameters.

### FilterElements

In addition to filtering entire documents, we also provide a convenience method for filtering elements from each Document in a DocSet. In this case, we supply a predicate that takes in elements and returns whether the element should be retained in the document. For example, if we aren't interested in processing images in our documents, we could filter them out with

```python
docset.filter_elements(lambda el: el.type != "Image")
```
