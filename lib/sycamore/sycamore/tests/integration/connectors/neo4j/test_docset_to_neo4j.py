from pydantic import BaseModel
import sycamore
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.extract_document_structure import StructureBySection
from sycamore.transforms.extract_graph_entities import EntityExtractor
from sycamore.transforms.extract_graph_relationships import RelationshipExtractor


def test_to_neo4j():
    ## actual test ##
    path = str(TEST_DIR / "resources/data/pdfs/doctor_testimonial.pdf")
    context = sycamore.init()
    llm = OpenAI(OpenAIModels.GPT_4O_MINI)

    URI = "neo4j://localhost:7687"
    AUTH = None
    DATABASE = "neo4j"

    class Doctor(BaseModel):
        name: str

    class Feedback(BaseModel):
        rating: int

    class MarketingMessage(BaseModel):
        drug_name: str
        message: str

    class Rated(BaseModel):
        start: Feedback
        end: MarketingMessage

    class Gave(BaseModel):
        start: Doctor
        end: Feedback

    ds = (
        context.read.binary(path, binary_format="pdf")
        .partition(
            partitioner=ArynPartitioner(extract_table_structure=True, use_ocr=True, extract_images=True),
            num_gpus=0.2,
        )
        .extract_document_structure(structure=StructureBySection)
        .extract_graph_entities([EntityExtractor(llm=llm, entities=[Doctor, Feedback, MarketingMessage])])
        .extract_graph_relationships([RelationshipExtractor(llm=llm, relationships=[Rated, Gave])])
        .resolve_graph_entities(resolvers=[])
        .explode()
    )
    ds.write.neo4j(uri=URI, auth=AUTH, database=DATABASE, import_dir="/neo4j/import")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(URI, auth=AUTH)
    session = driver.session(database=DATABASE)

    query1 = """
    MATCH (n:Doctor)
    RETURN COUNT(n) AS numDoctors
    """

    query2 = """
    MATCH (n:Feedback)
    RETURN COUNT(n) AS numFeedback
    """

    query3 = """
    MATCH (n:MarketingMessage)
    RETURN COUNT(n) AS numMessages
    """

    query4 = """
    MATCH (d:Doctor)-[r]->(mr:MarketingMessage)
    RETURN count(r) AS relationshipCount
    """

    query5 = """
    MATCH (d:Doctor)-[r]->(f:Feedback)
    RETURN count(r) AS relationshipCount
    """

    query6 = """
    MATCH (re:Feedback)-[r]->(mr:MarketingMessage)
    RETURN count(r) AS relationshipCount
    """

    # test nodes
    assert session.run(query1).single()["numDoctors"] == 1
    assert session.run(query2).single()["numFeedback"] == 2
    assert session.run(query3).single()["numMessages"] == 2

    # test relationships
    assert session.run(query4).single()["relationshipCount"] == 0
    assert session.run(query5).single()["relationshipCount"] == 2
    assert session.run(query6).single()["relationshipCount"] == 2

    session.close()
    driver.close()
