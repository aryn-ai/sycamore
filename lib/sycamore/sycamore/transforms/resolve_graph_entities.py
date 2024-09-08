from abc import ABC, abstractmethod
import json
from typing import TYPE_CHECKING, Any, Dict
import copy


from sycamore.data.document import Document, HierarchicalDocument, MetadataDocument
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
import logging

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
                            assert rel is not None, "relation value is None"
                            assert rel_uuid not in _nodes[0]["relationships"], "UUID Collision"
                            _nodes[0]["relationships"][rel_uuid] = rel
                    _nodes = [_nodes[0]]

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
       

    class AggregateSectionNodes(Map):
        def __init__(self, child: Node, **resource_args):
            super().__init__(child, f=self._aggregate_section_nodes, **resource_args)

        @staticmethod
        def _aggregate_section_nodes(doc: HierarchicalDocument) -> HierarchicalDocument:
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
    def _why_work(dataset: Any) -> Dict[str, Any]:
        docs_serialized = dataset.take_all()
        #docs = [Document.deserialize(doc["doc"]) for doc in docs_serialized]
        nodes = {}
        for row in docs_serialized:
            extracted = extract_nodes(row)
            merge_nodes(nodes, extracted)

        
        return nodes


    @staticmethod
    def _aggregate_document_nodes(dataset: Any) -> Dict[str, Any]:
        from ray.data.aggregate import AggregateFn

        def accumulate_row(nodes, row):
            extracted = extract_nodes(row)
#            logging.error(f"INPUT ACCUMULATE: {json.dumps(nodes, indent=2)}")
#            logging.error(f"INPUT ACCUMULATE: {json.dumps(extracted, indent=2)}")
            nodes = copy.deepcopy(nodes)
            check_if_bad(nodes)
            check_if_bad(extracted)
            merge_nodes(nodes, extracted)
            check_if_bad(nodes)
#            logging.error(f"OUTPUT ACCUMULATE: {json.dumps(nodes, indent=2)}")


            # for label, hashes in extracted.items():
            #     for hash, _nodes in hashes.items():
            #         nodes.setdefault(label, {})
            #         nodes[label].setdefault(hash, [])
            #         nodes[label][hash].extend(extracted[label][hash])
            return copy.deepcopy(nodes)

        def merge(nodes1_in, nodes2_in):
            nodes1 = copy.deepcopy(nodes1_in)
            nodes2 = copy.deepcopy(nodes2_in)
#            logging.error(f"INPUT MERGE: {json.dumps(nodes1, indent=2)}")
#            logging.error(f"INPUT MERGE: {json.dumps(nodes2, indent=2)}")
            check_if_bad(nodes1)
            check_if_bad(nodes2)
            merge_nodes(nodes1, nodes2)
            check_if_bad(nodes1)
#            logging.error(f"OUTPUT MERGE: {json.dumps(nodes1, indent=2)}")
            # for label, hashes in nodes2.items():
            #     for hash, _nodes in hashes.items():
            #         if label not in nodes1:
            #             nodes1[label] = dict()
            #         if hash not in nodes1[label]:
            #             nodes1[label][hash] = list()
            #         for node in _nodes:
            #             nodes1[label][hash].append(node)x   
            return copy.deepcopy(nodes1)

        def finalize(nodes):
            check_if_bad(nodes)
            return copy.deepcopy(nodes)

        aggregation = AggregateFn(
            init=lambda group_key: dict(), accumulate_row=accumulate_row, merge=merge, finalize=finalize, name="nodes"
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
            for child in doc.children:
                if "nodes" in child["properties"]:
                    del child["properties"]["nodes"]
        return doc


def extract_nodes(row):
    doc = Document.deserialize(row["doc"])
    if isinstance(doc, MetadataDocument) or "nodes" not in doc["properties"]:
        return dict()
    ret = {}
    if "Aircraft" not in doc["properties"]["nodes"]:
        return {}
    if "a55bd63b5d420fb34140005d40f0c5df0b2acb7a92ea93995793da8012880d00" not in doc["properties"]["nodes"]["Aircraft"]:
        return {}
    #docs_aircraft = [d for d in doc["properties"]["nodes"]["Aircraft"]["a55bd63b5d420fb34140005d40f0c5df0b2acb7a92ea93995793da8012880d00"] if d["doc_id"] == "7da8bbe9-1d43-49bd-9fe9-9ed3cc5e1a6e"]
    #if len(docs_aircraft) == 0:
    #    return {}
    
    #return {"Aircraft": {"a55bd63b5d420fb34140005d40f0c5df0b2acb7a92ea93995793da8012880d00": {"7da8bbe9-1d43-49bd-9fe9-9ed3cc5e1a6e": docs_aircraft}}}
    return {"Aircraft": {"a55bd63b5d420fb34140005d40f0c5df0b2acb7a92ea93995793da8012880d00": doc["properties"]["nodes"]["Aircraft"]["a55bd63b5d420fb34140005d40f0c5df0b2acb7a92ea93995793da8012880d00"] }}

    return doc["properties"]["nodes"]

def merge_nodes(nodes_to, nodes_from):
    for label, hashes in nodes_from.items():
        for hash, _nodes in hashes.items():
            nodes_to.setdefault(label, {})
            nodes_to[label].setdefault(hash, [])
            nodes_to[label][hash].extend(nodes_from[label][hash])

def check_if_bad(nodes):
    logging.error("ERIC CHECK IF BAD")
    for label, hashes in nodes.items():
        for hash, _nodes in hashes.items():
            for node in _nodes:
                for rel_uuid, rel in node["relationships"].items():
                    assert isinstance(rel, dict), f"{json.dumps(node['relationships'], indent=2)}"
