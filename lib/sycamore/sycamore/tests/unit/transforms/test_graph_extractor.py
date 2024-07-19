import sycamore
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_graph import GraphMetadata, MetadataExtractor
from sycamore.data import Document
from collections import defaultdict


class TestGraphExtractor:
    docs = [
        Document(
            {
                "doc_id": "1",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "3M", "sector": "Industrial", "doctype": "10K"},
                "elements": [],
            }
        ),
        Document(
            {
                "doc_id": "2",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "FedEx", "sector": "Industrial", "doctype": "10K"},
                "elements": [],
            }
        ),
        Document(
            {
                "doc_id": "3",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "Apple", "sector": "Technology", "doctype": "10K"},
                "elements": [],
            }
        ),
    ]

    def test_graph_extractor(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        metadata = [
            GraphMetadata(nodeKey="company", nodeLabel="Company", relLabel="FILED_BY"),
            GraphMetadata(nodeKey="sector", nodeLabel="Sector", relLabel="IN_SECTOR"),
            GraphMetadata(nodeKey="doctype", nodeLabel="Document Type", relLabel="IS_TYPE"),
        ]

        ds = ds.extract_graph_structure([MetadataExtractor(metadata=metadata)]).explode()
        docs = ds.take_all()

        nested_dict = defaultdict(lambda: defaultdict(list))

        for entry in docs:
            label = entry["label"]
            properties = entry["properties"]
            relations = entry["relationships"]

            for value in properties.values():
                for rel in relations.values():
                    nested_dict[label][value].append(rel)

        nested_dict = {label: dict(properties) for label, properties in nested_dict.items()}

        assert len(nested_dict["Company"]["3M"]) == 1
        assert nested_dict["Company"]["3M"][0]["START_ID"] == "1"
        assert len(nested_dict["Company"]["Apple"]) == 1
        assert nested_dict["Company"]["Apple"][0]["START_ID"] == "3"
        assert len(nested_dict["Company"]["FedEx"]) == 1
        assert nested_dict["Company"]["FedEx"][0]["START_ID"] == "2"
        assert len(nested_dict["Document Type"]["10K"]) == 3
        assert len(nested_dict["Sector"]["Industrial"]) == 2
        assert len(nested_dict["Sector"]["Technology"]) == 1
