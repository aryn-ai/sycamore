from abc import ABC, abstractmethod
import json
from typing import TYPE_CHECKING, Any, Dict

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
        docset.plan = self.AggregateSectionNodes(docset.plan)
        dataset = docset.plan.execute().materialize()
        # Perform ray aggregate over dataset
        nodes = self._aggregate_document_nodes(dataset)

        if self.resolve_duplicates:
            remap = {}
            for key, hashes in nodes.items():
                for hash, _nodes in hashes.items():
                    for i in range(1, len(_nodes)):
                        remap[_nodes[i]["doc_id"]] = _nodes[0]["doc_id"]
                        for rel_uuid, rel in _nodes[i]["relationships"].items():
                            assert rel_uuid not in _nodes[0]["relationships"], "UUID Collision"
                            _nodes[0]["relationships"][rel_uuid] = rel
                    _nodes = [_nodes[0]]

            for key, hashes in nodes.items():
                for hash, _nodes in hashes.items():
                    for node in _nodes:
                        existing_rels = set()
                        for rel_uuid, rel in node["relationships"].items():
                            rel["START_ID"] = remap.get(rel["START_ID"], rel["START_ID"])
                            rel["END_ID"] = remap.get(rel["END_ID"], rel["END_ID"])
                            if json.dumps(rel) in existing_rels:
                                del node["relationships"][rel_uuid]
                            else:
                                existing_rels.add(json.dumps(rel))

        for resolver in self.resolvers:
            nodes = resolver.resolve(nodes)

        # Load entities into a document, and add them to current ray dataset
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

    class AggregateSectionNodes(Map):
        def __init__(self, child: Node, **resource_args):
            super().__init__(child, f=self._aggregate_section_nodes, **resource_args)

        @staticmethod
        def _aggregate_section_nodes(doc: HierarchicalDocument) -> HierarchicalDocument:
            nodes: dict[str, Any] = {}
            for label, hashes in doc["properties"].get("nodes", {}).items():
                for hash, node in hashes.items():
                    nodes.setdefault(label, {})
                    nodes[label].setdefault(hash, [])
                    nodes[label][hash].append(node)
            for section in doc.children:
                for label, hashes in section["properties"].get("nodes", {}).items():
                    for hash, node in hashes.items():
                        nodes.setdefault(label, {})
                        nodes[label].setdefault(hash, [])
                        nodes[label][hash].append(node)
                del section["properties"]["nodes"]
            doc["properties"]["nodes"] = nodes
            return doc

    @staticmethod
    def _aggregate_document_nodes(dataset: Any) -> Dict[str, Any]:
        from ray.data.aggregate import AggregateFn

        def extract_nodes(row):
            doc = Document.deserialize(row["doc"])
            if isinstance(doc, MetadataDocument) or "nodes" not in doc["properties"]:
                return {}
            return doc["properties"]["nodes"]

        def accumulate_row(nodes, row):
            extracted = extract_nodes(row)
            for label, hashes in extracted.items():
                for hash in hashes:
                    nodes.setdefault(label, {})
                    nodes[label].setdefault(hash, [])
                    nodes[label][hash].extend(extracted[label][hash])
            return nodes

        def merge(nodes1, nodes2):
            for label, hashes in nodes2.items():
                for hash in hashes:
                    nodes1.setdefault(label, {})
                    nodes1[label].setdefault(hash, [])
                    nodes1[label][hash].extend(nodes2[label][hash])
            return nodes1

        def finalize(nodes):
            return nodes

        aggregation = AggregateFn(
            init=lambda group_key: {}, accumulate_row=accumulate_row, merge=merge, finalize=finalize, name="nodes"
        )

        result = dataset.aggregate(aggregation)["nodes"]
        return result


class CleanTempNodes(Map):
    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=self._clean_temp_nodes, **resource_args)

    @staticmethod
    def _clean_temp_nodes(doc: HierarchicalDocument) -> HierarchicalDocument:
        if "properties" in doc and "nodes" in doc["properties"]:
            del doc["properties"]["nodes"]
        return doc
