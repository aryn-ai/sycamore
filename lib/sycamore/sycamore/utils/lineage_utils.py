from sycamore.data import Document, MetadataDocument


def update_lineage(from_docs: list[Document], to_docs: list[Document]) -> list[MetadataDocument]:
    from_ids = [d.lineage_id for d in from_docs]
    for d in to_docs:
        d.update_lineage_id()
    to_ids = [d.lineage_id for d in to_docs]

    return [MetadataDocument(lineage_links={"from_ids": from_ids, "to_ids": to_ids})]
