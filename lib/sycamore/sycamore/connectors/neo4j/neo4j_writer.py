from dataclasses import dataclass
from collections import defaultdict
import json
import typing
from typing import Any, Union

from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.data.document import Document, MetadataDocument

import time
import os
import csv
import logging
import urllib

from sycamore.plan_nodes import Node, Write
from sycamore.transforms.map import MapBatch
from sycamore.utils.time_trace import TimeTrace
from sycamore.utils.import_utils import requires_modules

import fcntl

if typing.TYPE_CHECKING:
    from neo4j import Auth, Driver, Session
    from neo4j.auth_management import AuthManager

logger = logging.getLogger(__name__)


@dataclass
class Neo4jWriterTargetParams(BaseDBWriter.TargetParams):
    database: str


@dataclass
class Neo4jWriterClientParams(BaseDBWriter.TargetParams):
    uri: str
    auth: Union[tuple[Any, Any], "Auth", "AuthManager", None]
    import_dir: str


class Neo4jWriterClient:
    def __init__(self, driver: "Driver", import_dir: str):
        self._driver = driver
        self._import_dir = import_dir

    @classmethod
    @requires_modules("neo4j", extra="neo4j")
    def from_client_params(cls, params: Neo4jWriterClientParams) -> "Neo4jWriterClient":
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri=params.uri, auth=params.auth)
        driver.verify_connectivity()
        return Neo4jWriterClient(driver, params.import_dir)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, Neo4jWriterTargetParams)
        with self._driver.session(database=target_params.database):
            pass

    def _write_nodes_neo4j(self, nodes: list[str], session: "Session"):
        for node_type in nodes:
            file_url = f"file:///sycamore/nodes/{urllib.parse.quote(node_type)}"
            node_label = node_type[:-4]
            build_nodes = f"""
            LOAD CSV WITH HEADERS FROM "{file_url}" AS row

            MERGE (n:`{node_label}` {{uuid: row["uuid:ID"]}})

            WITH n, row, [key IN keys(row) WHERE key <> "uuid:ID"] AS keys

            CALL apoc.create.setProperties(n, keys, [key IN keys | row[key]]) YIELD node
            RETURN node;
            """
            with session.begin_transaction() as tx:
                tx.run(build_nodes)
        # clean up delete csv's
        for node_type in nodes:
            path = self._import_dir + f"/sycamore/nodes/{node_type}"
            if os.path.exists(path):
                os.remove(path)
            else:
                logger.warn(f"ERROR: {path} does not exist, cannot delete")

    def _write_relationships_neo4j(self, relationships: list[str], session: "Session"):
        for relationship_type in relationships:
            file_url = f"file:///sycamore/relationships/{urllib.parse.quote(relationship_type)}"
            start_label, end_label = (relationship_type[:-4]).split("_")
            build_relationships = f"""
            CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "{file_url}" AS row RETURN row',
            '
            MATCH (s:`{start_label}` {{uuid: row[":START_ID"]}})
            MATCH (e:`{end_label}` {{uuid: row[":END_ID"]}})
            WITH s, e, row, apoc.map.removeKeys(row, [":START_ID", ":END_ID", ":TYPE", "uuid:ID"]) AS properties
            CALL apoc.create.relationship(
              s,
              row[":TYPE"],
              apoc.map.merge(properties, {{uuid: row["uuid:ID"]}}),
              e
            ) YIELD rel
            RETURN rel
            ',
            {{batchSize: 2500, parallel: true}})
            """
            with session.begin_transaction() as tx:
                tx.run(build_relationships)
        # clean up delete csv's
        for relationship_type in relationships:
            path = self._import_dir + f"/sycamore/relationships/{relationship_type}"
            if os.path.exists(path):
                os.remove(path)
            else:
                logger.warn(f"ERROR: {path} does not exist, cannot delete")

    def _write_constraints_neo4j(self, labels: list[str], session: "Session"):
        for node_label in labels:
            query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{node_label}`) REQUIRE n.uuid IS UNIQUE;"
            with session.begin_transaction() as tx:
                tx.run(query)

    def write_to_neo4j(
        self, nodes: list[str], relationships: list[str], labels: list[str], target_params: BaseDBWriter.TargetParams
    ):
        assert isinstance(target_params, Neo4jWriterTargetParams)
        with self._driver.session(database=target_params.database) as session:
            start = time.time()
            self._write_constraints_neo4j(labels, session)
            self._write_nodes_neo4j(nodes, session)
            self._write_relationships_neo4j(relationships, session)
            end = time.time()
            logger.info(f"TIME TAKEN TO LOAD CSV --> NEO4J: {end-start} SECONDS")


class Neo4jValidateParams:
    def __init__(self, client_params: Neo4jWriterClientParams, target_params: Neo4jWriterTargetParams):
        self._client = Neo4jWriterClient.from_client_params(client_params)
        self._client.create_target_idempotent(target_params=target_params)
        self._check_write_permissions(client_params=client_params)
        self._client._driver.close()

    def _check_write_permissions(self, client_params: Neo4jWriterClientParams):
        path = client_params.import_dir
        # Check read permissions
        if not os.access(path, os.R_OK):
            raise OSError(f"Read permission denied for directory: {path}")
        # Check write permissions
        if not os.access(path, os.W_OK):
            raise OSError(f"Write permission denied for directory: {path}")


class Neo4jPrepareCSV:
    def __init__(self, plan: Node, client_params: Neo4jWriterClientParams):
        self._dataset = plan.execute()
        self._import_dir = client_params.import_dir

        res = self.aggregate()
        nodes = res["nodes"]
        relationships = res["relationships"]
        self._write_nodes_csv_headers(nodes)
        self._write_relationships_csv_headers(relationships)

    def aggregate(self) -> Any:
        from ray.data.aggregate import AggregateFn

        def extract_nodes(row):
            doc = Document.deserialize(row["doc"])
            if isinstance(doc, MetadataDocument):
                return {}
            return doc.data

        def accumulate_row(headers, row):
            include_nodes = ["type", "bbox", "text_representation"]
            include_relationships = ["uuid:ID", ":START_ID", ":END_ID", ":TYPE"]
            data = extract_nodes(row)
            if "label" not in data:
                return headers
            node_key = data["label"]
            headers["nodes"].setdefault(node_key, dict())
            #### ALL DOCS HAVE uuid:ID ####
            headers["nodes"][node_key]["uuid:ID"] = True
            #### IF KEYS EXIST IN DATA ####
            for key in include_nodes:
                if key in data:
                    headers["nodes"][node_key][key] = True
            #### add all keys from properties ####
            for key in data["properties"].keys():
                headers["nodes"][node_key][key] = True
            for key, value in data["relationships"].items():
                rel_key = value["START_LABEL"] + "_" + value["END_LABEL"]
                if headers["relationships"].get(rel_key, None) is None:
                    headers["relationships"][rel_key] = dict()
                #### add all required keys ####
                for key in include_relationships:
                    headers["relationships"][rel_key][key] = True
                #### add all keys from properties ####
                for key in value["properties"].keys():
                    headers["relationships"][rel_key][key] = True
            return headers

        def merge(headers1, headers2):
            #### merge nodes together ####
            for key, values in headers2["nodes"].items():
                headers1["nodes"].setdefault(key, dict())
                headers1["nodes"][key] |= values
            #### merge relationships together ####
            for key, values in headers2["relationships"].items():
                headers1["relationships"].setdefault(key, dict())
                headers1["relationships"][key] |= values
            return headers1

        def finalize(nodes):
            return nodes

        aggregation = AggregateFn(
            init=lambda group_key: dict(nodes=dict(), relationships=dict()),
            accumulate_row=accumulate_row,
            merge=merge,
            finalize=finalize,
            name="output",
        )

        return self._dataset.aggregate(aggregation)["output"]

    def _write_nodes_csv_headers(self, nodes):
        for node_label, node_columns in nodes.items():
            csv_path = os.path.expanduser(f"{self._import_dir}/sycamore/nodes/{node_label}.csv")
            headers = [column for column in node_columns.keys()]
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)

    def _write_relationships_csv_headers(self, relationships):
        for relationship_label, relationship_columns in relationships.items():
            csv_path = os.path.expanduser(f"{self._import_dir}/sycamore/relationships/{relationship_label}.csv")
            headers = [column for column in relationship_columns.keys()]
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)


class Neo4jWriteCSV(MapBatch, Write):
    def __init__(self, plan: Node, client_params: Neo4jWriterClientParams, **kwargs):
        self._import_dir = client_params.import_dir
        self._name = "Neo4jWriter"
        super().__init__(plan, f=self._write_docs_tt, **kwargs)

    @staticmethod
    def _parse_docs(docs):
        include_nodes = ["type", "bbox", "text_representation"]
        include_relationships = ["uuid:ID", ":START_ID", ":END_ID", ":TYPE"]
        nodes = defaultdict(list)
        relationships = defaultdict(lambda: defaultdict(list))
        for doc in docs:
            if "label" not in doc.data or "doc_id" not in doc.data:
                continue
            node = {
                "uuid:ID": doc.doc_id,
            }
            for key, value in doc.data.items():
                # add properties to node
                if key == "properties":
                    for property_key, property_value in value.items():
                        if property_key not in include_nodes:
                            if isinstance(property_value, list) or isinstance(property_value, dict):
                                node[property_key] = json.dumps(property_value)
                            else:
                                node[property_key] = property_value
                # add included fields to node
                if key in include_nodes:
                    node[key] = value
            nodes[doc.data["label"]].append(node)

            for key, value in doc.data["relationships"].items():
                rel = {
                    "uuid:ID": key,
                    ":START_ID": value["START_ID"],
                    ":END_ID": value["END_ID"],
                    ":TYPE": value["TYPE"],
                }
                for property_key, property_value in value["properties"].items():
                    if property_key not in include_relationships:
                        if isinstance(property_value, list) or isinstance(property_value, dict):
                            node[property_key] = json.dumps(property_value)
                        else:
                            node[property_key] = property_value
                relationships[value["START_LABEL"]][value["END_LABEL"]].append(rel)
        return nodes, relationships

    def _write_nodes_csv(self, nodes):
        for node_label, rels in nodes.items():
            csv_path = os.path.expanduser(f"{self._import_dir}/sycamore/nodes/{node_label}.csv")
            headers = None
            with open(csv_path, mode="r", newline="") as file:
                csv_reader = csv.reader(file)
                headers = list(next(csv_reader))

            with open(csv_path, "a", newline="") as file:
                # get exclusive lock on csv file
                fcntl.flock(file, fcntl.LOCK_EX)
                try:
                    writer = csv.writer(file)
                    for entry in rels:
                        row = [entry.get(header, "") for header in headers]
                        writer.writerow(row)
                finally:
                    # release exclusive lock on csv file
                    fcntl.flock(file, fcntl.LOCK_UN)

    def _write_relationships_csv(self, relationships):
        for start_label, end_labels in relationships.items():
            for end_label, rels in end_labels.items():
                csv_path = os.path.expanduser(
                    f"{self._import_dir}/sycamore/relationships/{start_label}_{end_label}.csv"
                )
                headers = None
                with open(csv_path, mode="r", newline="") as file:
                    csv_reader = csv.reader(file)
                    headers = list(next(csv_reader))

                with open(csv_path, "a", newline="") as file:
                    # get exclusive lock on csv file
                    fcntl.flock(file, fcntl.LOCK_EX)
                    try:
                        writer = csv.writer(file)
                        for entry in rels:
                            row = [entry.get(header, "") for header in headers]
                            writer.writerow(row)
                    finally:
                        # release exclusive lock on csv file
                        fcntl.flock(file, fcntl.LOCK_UN)

    def write_docs(self, docs: list[Document]) -> list[Document]:

        nodes, relationships = self._parse_docs(docs)
        self._write_nodes_csv(nodes)
        self._write_relationships_csv(relationships)
        return docs

    def _write_docs_tt(self, docs: list[Document]) -> list[Document]:
        if self._name:
            with TimeTrace(self._name):
                return self.write_docs(docs)
        else:
            with TimeTrace("UnknownWriter"):
                return self.write_docs(docs)


class Neo4jLoadCSV:
    def __init__(self, client_params: Neo4jWriterClientParams, target_params: Neo4jWriterTargetParams, **kwargs):
        self._client = Neo4jWriterClient.from_client_params(client_params)
        self._nodes = [f for f in os.listdir(client_params.import_dir + "/sycamore/nodes")]
        self._relationships = [f for f in os.listdir(client_params.import_dir + "/sycamore/relationships")]
        self._labels = [f[:-4] for f in self._nodes]  # f[:-4] used to drop '.csv'
        self._client.write_to_neo4j(self._nodes, self._relationships, self._labels, target_params)
        self._client._driver.close()
