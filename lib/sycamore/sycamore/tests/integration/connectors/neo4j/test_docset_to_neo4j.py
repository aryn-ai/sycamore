import sycamore
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import SycamorePartitioner


def test_to_neo4j():

    ## actual test ##
    path = str(TEST_DIR / "resources/data/pdfs/Ray_page11.pdf")
    context = sycamore.init()
    URI = "neo4j://localhost:7687"
    AUTH = ("neo4j", "koala-stereo-comedy-spray-figure-6974")
    DATABASE = "neo4j"

    ds = (
        context.read.binary(path, binary_format="pdf")
        .partition(
            partitioner=SycamorePartitioner(extract_table_structure=True, use_ocr=True, extract_images=True),
            num_gpus=0.2,
        )
        .extract_graph_structure(extractors=[])
        .explode()
    )

    ds.write.neo4j(uri=URI, auth=AUTH, database=DATABASE, import_dir="/home/admin/neo4j/import")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(URI, auth=AUTH)
    session = driver.session(database=DATABASE)

    query1 = """
    MATCH (n:ELEMENT {type: 'table'})
    RETURN COUNT(n) AS numTables
    """
    query2 = """
    MATCH (n:DOCUMENT)
    RETURN COUNT(n) AS numDocuments
    """
    query3 = """
    MATCH (n:SECTION)
    RETURN COUNT(n) AS numSections
    """
    assert session.run(query1).single()["numTables"] == 2
    assert session.run(query2).single()["numDocuments"] == 1
    assert session.run(query3).single()["numSections"] == 4

    session.close()
    driver.close()
