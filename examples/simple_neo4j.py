import sycamore
from sycamore.transforms.extract_document_structure import StructureByDocument
from sycamore.transforms.extract_graph_entities import EntityExtractor
from sycamore.transforms.extract_graph_relationships import RelationshipExtractor
from sycamore.transforms.partition import ArynPartitioner

from sycamore.llms import OpenAI, OpenAIModels

from typing import Optional
from pydantic import BaseModel, Field

llm = OpenAI(OpenAIModels.GPT_4O_STRUCTURED)
ctx = sycamore.init()


class Report(BaseModel):
    ID: str = Field(
        description="This ID can be found under either Accident Number, Incident Number, or Occurance Number"
    )


class Aircraft(BaseModel):
    registration: str
    make: str
    model: str


class Pilot(BaseModel):
    is_student: bool
    total_flight_hours: Optional[int]


class INVOLVED_AIRCRAFT(BaseModel):
    start: Report
    end: Aircraft


class FLOWN_BY(BaseModel):
    start: Aircraft
    end: Pilot


entity_extractors = [EntityExtractor(llm=llm, entities=[Report, Aircraft, Pilot])]

relationship_extractors = [RelationshipExtractor(llm=llm, relationships=[INVOLVED_AIRCRAFT, FLOWN_BY])]

paths = "s3://aryn-public/knowledge-graph-blog-data/"
ds = (
    ctx.read.binary(paths=paths, binary_format="pdf")
    .partition(partitioner=ArynPartitioner(extract_table_structure=True, use_ocr=True, extract_images=True))
    .extract_document_structure(StructureByDocument)
    .extract_graph_entities(entity_extractors)
    .extract_graph_relationships(relationship_extractors)
    .resolve_graph_entities(resolve_duplicates=True)
    .explode()
)

############################################################################
# TO SETUP NEO4J LOCALLY, READ THE DOCS:                                   #
# https://sycamore.readthedocs.io/en/stable/sycamore/connectors/neo4j.html #
############################################################################

URI = "<ENTER_NEO4J_URI>"  # default: "bolt://localhost:7687"
AUTH = ("<ENTER_NEO4J_USERNAME>", "<ENTER_NEO4J_PASSWORD>")
DATABASE = "<ENTER_NEO4J_DB>"  # default: "neo4j"
IMPORT_DIR = "<ENTER_NEO4J_IMPORT_DIR>"  # configured during neo4j docker run
ds.write.neo4j(uri=URI, auth=AUTH, import_dir=IMPORT_DIR, database=DATABASE)
