## FlatMap
The FlatMap transform takes a function from a single ``Document`` to a list of ``Documents``, and returns then "flattens" the result into a single ``DocSet``. In the following example, the FlatMap transform outputs a new list of documents
where each document includes elements from a single page only.
```python
def split_and_convert_to_image(doc: Document) -> list[Document]:
    if doc.binary_representation is not None:
        images = pdf2image.convert_from_bytes(doc.binary_representation)
    else:
        return [doc]

    elements_by_page: dict[int, list[Element]] = {}

    for e in doc.elements:
        page_number = e.properties["page_number"]
        elements_by_page.setdefault(page_number, []).append(e)

    new_docs = []
    for page, elements in elements_by_page.items():
        new_doc = Document(elements={"array": elements})
        new_doc.properties.update(doc.properties)
        new_doc.properties.update({"page_number": page})
        new_docs.append(new_doc)
    return new_docs

docset = docset.flat_map(split_and_convert_to_image)
```