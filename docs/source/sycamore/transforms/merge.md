## Merge

The merge transform is responsible for 'merging' elements into larger 'chunks'. This is also known as 'chunking.'

The merge transform takes a single argument -- the `merger`, which contains the logic defining which elements to merge and how to merge them. The available mergers are listed below. More information can be found in the {doc}`API documentation </sycamore/APIs/low_level_transforms/merge_elements>`

### Greedy Text Element Merger

The `GreedyTextElementMerger` takes a tokenizer and a token limit, and merges elements together, greedily, until the combined element will overflow the token limit, at which point the merger starts work on a new merged element. If an element is already too big, the `GreedyTextElementMerger` will leave it alone.

For example, using a `CharacterTokenizer` with `max_tokens=4`, you would have the following behavior:

```
A BC D EF GHI J KLMNO -> ABCD EF GHIJ KLMNO
```

To add this to a script:

```python
merger = GreedyTextElementMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2", max_tokens=512))
merged_docset = docset.merge(merger=merger)
```

### Greedy Section Merger

I'm a little confused by this explanation. I might say: Like the `GreedyTextElementMerger`, the `GreedySectionMerger` takes a tokenizer and token limit, and merges elements greedily, stopping before the token limit. However, the `GreedySectionMerger` also has additional heuristic rules.

Use it in much the same way as the text element merger:

```python
merger = GreedySectionMerger(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2", max_tokens=512))
merged_docset = docset.merge(merger=merger)
```

### Marked Merger

The `MarkedMerger` merges elements by referencing "marks" placed on the elements by the transforms {doc}`here </sycamore/APIs/low_level_transforms/mark_misc>` and {doc}`here </sycamore/APIs/low_level_transforms/bbox_merge>`.
The marks are "_break" and "_drop". The `MarkedMerger` will merge elements until it hits a "_break" mark, whereupon it will start a new element. It handles elements marked with "_drop" by, well, dropping them entirely. This merger is useful when you have many rules to apply to how you want to chunk your document.

We have found that the `MarkedMerger` is best used with the DocSet method `docset.mark_bbox_preset`, which applies a pre-defined series of marking transforms.

```python
marked_ds = docset.mark_bbox_preset(tokenizer=HuggingFaceTokenizer("sentence-transformers/all-MiniLM-L6-v2", token_limit=512))
merged_ds = marked_ds.merge(merger=MarkedMerger())
```
