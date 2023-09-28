## Embed
The Embed Transform is responsible for generating embeddings for your Documents or Elements. These embeddings are stored in a special ``embedding`` property on each document.
The initial embedding implementation is the ``SentenceTransformerEmbedder``, which embeds the text representation of each document using any of the models from the popular [SentenceTransformers framework](https://www.sbert.net/). For example, the following code embeds a ``DocSet`` with the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model:

```python
embedder = SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
embedded_doc_set = docset.embed(embedder)
```

During execution, Sycamore will automatically batch records and leverage GPUs where appropriate.
