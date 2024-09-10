from abc import ABC, abstractmethod
import json
from typing import TYPE_CHECKING, Any, Dict
import copy


from sycamore.data.document import Document, HierarchicalDocument, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map

if TYPE_CHECKING:
    from sycamore.docset import DocSet


class EntityResolver(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def resolve(self, entities: Any) -> Any:
        pass


class ResolveEntities:
    """
    Groups entity nodes and their respective relationships together into a document
    where we run through all the Entity Resolution proceedures defined by the user.
    """

    def __init__(self, resolvers: list[EntityResolver], resolve_duplicates):
        self.resolvers = resolvers
        self.resolve_duplicates = resolve_duplicates

    def resolve(self, docset: "DocSet") -> Any:
        from ray.data import from_items

        # Group nodes from document sections together and materialize docset into ray dataset
        docset.plan = self.MergeSectionNodes(docset.plan)
        dataset = docset.plan.execute().materialize()
        # Perform ray aggregate over dataset
        nodes = self._merge_document_nodes(dataset)

        if self.resolve_duplicates:
            remap = {}
            for key, hashes in nodes.items():
                for hash, _nodes in hashes.items():
                    for i in range(1, len(_nodes)):
                        remap[_nodes[i]["doc_id"]] = _nodes[0]["doc_id"]
                        for rel_uuid, rel in _nodes[i]["relationships"].items():
                            assert rel is not None, "relation value is None"
                            assert rel_uuid not in _nodes[0]["relationships"], "UUID Collision"
                            _nodes[0]["relationships"][rel_uuid] = rel
                    hashes[hash] = [_nodes[0]]

            for key, hashes in nodes.items():
                for hash, _nodes in hashes.items():
                    for node in _nodes:
                        remove = []
                        existing_rels = set()
                        for rel_uuid, rel in node["relationships"].items():
                            rel["START_ID"] = remap.get(rel["START_ID"], rel["START_ID"])
                            rel["END_ID"] = remap.get(rel["END_ID"], rel["END_ID"])
                            if json.dumps(rel) in existing_rels:
                                remove.append(rel_uuid)
                            else:
                                existing_rels.add(json.dumps(rel))
                        for r in remove:
                            del node["relationships"][r]

        for resolver in self.resolvers:
            nodes = resolver.resolve(nodes)

        doc = HierarchicalDocument()
        for key, hashes in nodes.items():
            for hash, nodes in hashes.items():
                for node in nodes:
                    node = HierarchicalDocument(node)
                    doc.children.append(node)
        doc.data["EXTRACTED_NODES"] = True
        nodes_row = from_items([{"doc": doc.serialize()}])
        dataset = dataset.union(nodes_row)

        return dataset

        # Load entities into a document, and add them to current ray dataset

    class MergeSectionNodes(Map):
        def __init__(self, child: Node, **resource_args):
            super().__init__(child, f=self._merge_section_nodes, **resource_args)

        @staticmethod
        def _merge_section_nodes(doc: HierarchicalDocument) -> HierarchicalDocument:
            nodes: dict[str, Any] = {}
            for section in doc.children:
                for label, hashes in section["properties"].get("nodes", {}).items():
                    for hash, node in hashes.items():
                        nodes.setdefault(label, {})
                        nodes[label].setdefault(hash, [])
                        nodes[label][hash].append(node)
            doc["properties"]["nodes"] = copy.deepcopy(nodes)
            return doc

    @staticmethod
    def _merge_document_nodes(dataset: Any) -> Dict[str, Any]:
        docs_serialized = dataset.take_all()
        nodes: dict[str, Any] = {}
        for row in docs_serialized:
            extracted = extract_nodes(row)
            merge_nodes(nodes, extracted)
        check_null_relationships(nodes)
        return nodes


class CleanTempNodes(Map):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=self._clean_temp_nodes, **resource_args)

    @staticmethod
    def _clean_temp_nodes(doc: HierarchicalDocument) -> HierarchicalDocument:
        if "properties" in doc and "nodes" in doc["properties"]:
            del doc["properties"]["nodes"]
            for child in doc.children:
                if "nodes" in child["properties"]:
                    del child["properties"]["nodes"]
        return doc


def extract_nodes(row):
    doc = Document.deserialize(row["doc"])
    if isinstance(doc, MetadataDocument) or "nodes" not in doc["properties"]:
        return dict()

    return doc["properties"]["nodes"]


def merge_nodes(nodes_to, nodes_from):
    for label, hashes in nodes_from.items():
        for hash, _nodes in hashes.items():
            nodes_to.setdefault(label, {})
            nodes_to[label].setdefault(hash, [])
            nodes_to[label][hash].extend(nodes_from[label][hash])


def check_null_relationships(nodes):
    for label, hashes in nodes.items():
        for hash, _nodes in hashes.items():
            for node in _nodes:
                for rel_uuid, rel in node["relationships"].items():
                    assert isinstance(rel, dict), f"{json.dumps(node['relationships'], indent=2)}"
