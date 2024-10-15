from pathlib import Path

from sycamore.data import Document, TableElement
from sycamore.data.document import DocumentPropertyTypes


def to_html_tables(doc: Document) -> list[Document]:
    new_docs = []
    for table_num, e in enumerate(el for el in doc.elements if el.type == "table"):
        if not isinstance(e, TableElement) or e.table is None:
            raise ValueError(f"Unable to generate html string for element {e}")

        new_text = e.table.to_html(pretty=True, wrap_in_html=True)

        new_doc = Document(text_representation=new_text)

        new_doc.properties["path"] = doc.properties["path"]

        if DocumentPropertyTypes.PAGE_NUMBER in doc.properties:
            new_doc.properties[DocumentPropertyTypes.PAGE_NUMBER] = doc.properties[DocumentPropertyTypes.PAGE_NUMBER]
        new_doc.properties["table_num"] = table_num
        new_docs.append(new_doc)

    return new_docs


def html_table_filename_fn(doc: Document) -> str:
    path = Path(doc.properties["path"])
    base_name = ".".join(path.name.split(".")[0:-1])
    if "table_num" in doc.properties:
        suffix = doc.properties["table_num"]
    else:
        suffix = doc.doc_id
    return f"{base_name}_table_{suffix}.html"
