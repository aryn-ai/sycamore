## Explode

The Explode transform converts the elements of each document into top-level documents. For example, if you explode a ``DocSet`` with a single document containing two elements, the resulting ``DocSet`` will have three documents -- the original plus a new ``Document`` for each of the elements.

```python
exploded_doc_set = docset.explode()
```

The primary use of the explode transform is to embed and ingest chunks of your document, the elements, as independent records in a data store like OpenSearch.