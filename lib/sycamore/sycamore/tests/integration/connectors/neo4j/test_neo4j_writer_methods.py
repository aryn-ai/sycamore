from sycamore.connectors.neo4j import Neo4jWriteCSV
from sycamore.data import HierarchicalDocument

def test_parse_docs():
    docs = [
            HierarchicalDocument(
                {
                    "doc_id": "1",
                    "label": "DOCUMENT",
                    "type": "pdf",
                    "relationships": {
                        "100": {"START_ID": "1", "START_LABEL": "DOCUMENT", "END_ID": "2",
                                "END_LABEL": "DOCUMENT", "TYPE": "RELATED_TO", "properties": {}}
                    },
                    "properties": {},
                    "children": [],
                }
            ),
            HierarchicalDocument(
                {
                    "doc_id": "2",
                    "label": "DOCUMENT",
                    "type": "pdf",
                    "relationships": {
                        "101": {"START_ID": "2", "START_LABEL": "DOCUMENT", "END_ID": "3",
                                "END_LABEL": "DOCUMENT", "TYPE": "RELATED_TO", "properties": {}}
                    },
                    "properties": {},
                    "children": [],
                }
            ),
            HierarchicalDocument(
                {
                    "doc_id": "3",
                    "label": "DOCUMENT",
                    "type": "pdf",
                    "relationships": {
                        "102": {"START_ID": "3", "START_LABEL": "DOCUMENT", "END_ID": "2",
                                "END_LABEL": "DOCUMENT", "TYPE": "RELATED_TO", "properties": {}}
                    },
                    "properties": {},
                    "children": [],
                }
            ),
        ]

    nodes, relationships = Neo4jWriteCSV._parse_docs(docs)

    nodes_expected = {
        "DOCUMENT": [
            {"uuid:ID": "1", "type": "pdf"},
            {"uuid:ID": "2", "type": "pdf"},
            {"uuid:ID": "3", "type": "pdf"}
        ]
    }

    relationships_expected = {
        "DOCUMENT": {
            "DOCUMENT": [
                {"uuid:ID": "100", ":START_ID": "1", ":END_ID": "2", ":TYPE": "RELATED_TO"},
                {"uuid:ID": "101", ":START_ID": "2", ":END_ID": "3", ":TYPE": "RELATED_TO"},
                {"uuid:ID": "102", ":START_ID": "3", ":END_ID": "2", ":TYPE": "RELATED_TO"}
            ]
        }
    }

    assert nodes_expected == nodes
    assert relationships_expected == relationships
