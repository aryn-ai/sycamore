from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict
import uuid

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
    def __init__(self, resolvers: list[EntityResolver]):
        self.resolvers = resolvers

    def resolve(self, docset: "DocSet") -> Any:
        from ray.data import from_items

        # Group nodes from document sections together and materialize docset into ray dataset
        docset.plan = self.AggregateSectionNodes(docset.plan)
        dataset = docset.plan.execute().materialize()

        # Perform ray aggregate over dataset
        nodes = self._aggregate_document_nodes(dataset)
        for resolver in self.resolvers:
            nodes = resolver.resolve(nodes)

        # Load entities into a document, and add them to current ray dataset
        doc = HierarchicalDocument()
        for key, hashes in nodes.items():
            for hash, node in hashes.items():
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
            nodes: defaultdict[dict, Any] = defaultdict(dict)
            nodes |= doc["properties"].get("nodes", {})
            for section in doc.children:
                if "nodes" not in section["properties"]:
                    continue
                for label, hashes in section["properties"]["nodes"].items():
                    for hash, node in hashes.items():
                        if nodes[label].get(hash, None) is None:
                            nodes[label][hash] = node
                        else:
                            for rel_uuid, rel in node["relationships"].items():
                                if rel not in [node for node in nodes[label][hash]["relationships"].values()]:
                                    nodes[label][hash]["relationships"][rel_uuid] = rel
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
            for key, hashes in extracted.items():
                for hash in hashes:
                    if nodes.get(key, {}).get(hash, {}) == {}:
                        nodes.setdefault(key, {})
                        nodes[key][hash] = extracted[key][hash]
                    else:
                        for rel_uuid, rel in extracted[key][hash]["relationships"].items():
                            if rel not in [node for node in nodes[key][hash]["relationships"].values()]:
                                nodes[key][hash]["relationships"][rel_uuid] = rel
            return nodes

        def merge(nodes1, nodes2):
            for key, hashes in nodes2.items():
                for hash in hashes:
                    if nodes1.get(key, {}).get(hash, {}) == {}:
                        nodes1.setdefault(key, {})
                        nodes1[key][hash] = nodes2[key][hash]
                    else:
                        for rel_uuid, rel in nodes2[key][hash]["relationships"].items():
                            if rel not in [node for node in nodes1[key][hash]["relationships"].values()]:
                                nodes1[key][hash]["relationships"][rel_uuid] = rel
            return nodes1

        def finalize(nodes):
            for hashes in nodes.values():
                for node in hashes.values():
                    node["doc_id"] = str(uuid.uuid4())
                    for rel in node["relationships"].values():
                        rel["END_ID"] = node["doc_id"]
                        rel["END_LABEL"] = node["label"]

            for hashes in nodes.values():
                for node in hashes.values():
                    for rel in node["relationships"].values():
                        if "START_HASH" in rel:
                            rel["START_ID"] = nodes[rel["START_LABEL"]][rel["START_HASH"]]["doc_id"]
                            del rel["START_HASH"]
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
