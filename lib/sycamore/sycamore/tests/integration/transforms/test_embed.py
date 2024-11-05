import sycamore
from sycamore.data import Document
from sycamore.transforms.embed import Embedder, BedrockEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder

passages = [
    (
        "Abraham Lincoln (/ˈlɪŋkən/ LINK-ən; February 12, 1809 – April 15, 1865) was an American lawyer, politician,"
        "and statesman who served as the 16th president of the United States from 1861 until his assassination in 1865."
    ),
    (
        "During 1831 and 1832, Lincoln worked at a general store in New Salem, Illinois. In 1832, he declared his"
        "candidacy for the Illinois House of Representatives, but interrupted his campaign to serve as a captain in"
        "the Illinois Militia during the Black Hawk War.[59] When Lincoln returned home from the Black Hawk War,"
        " he planned to become a blacksmith, but instead formed a partnership with 21-year-old William Berry, with"
        "whom he purchased a New Salem general store on credit. Because a license was required to sell customers "
        "single beverages, Berry obtained bartending licenses for $7 each for Lincoln and himself, and in 1833 the"
        "Lincoln-Berry General Store became a tavern as well. As licensed bartenders, Lincoln and Berry were able"
        " to sell spirits, including liquor, for 12 cents a pint. They offered a wide range of alcoholic beverages"
        " as well as food, including takeout dinners. But Berry became an alcoholic, was often too drunk to work,"
        " and Lincoln ended up running the store by himself.[60] Although the economy was booming, the business"
        " struggled and went into debt, causing Lincoln to sell his share."
    ),
    (""),
]


def check_embedder(embedder: Embedder, expected_dim: int):
    docs = [
        Document(
            {
                "doc_id": f"doc_{i}",
                "type": "test",
                "text_representation": passage,
                "elements": [],
                "properties": {},
            }
        )
        for i, passage in enumerate(passages)
    ]

    new_docs = embedder.generate_embeddings(docs)
    assert len(new_docs) == len(docs)

    for doc in new_docs:
        if doc.text_representation != "":
            assert doc.embedding is not None
            assert len(doc.embedding) == expected_dim

def test_sentencetransformer_embedding():
    check_embedder(embedder=SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100), expected_dim=384)

def test_openai_embedding():
    check_embedder(embedder=OpenAIEmbedder(), expected_dim=1536)


def test_bedrock_embedding():
    check_embedder(embedder=BedrockEmbedder(), expected_dim=1536)


def test_openai_embedding_batches():
    docs = [
        Document(
            {
                "doc_id": f"doc_{i}",
                "type": "test",
                "text_representation": f"Document text for passage {i}",
                "elements": [],
                "properties": {},
            }
        )
        for i in range(5)
    ]

    context = sycamore.init()
    doc_set = context.read.document(docs)

    embedder = SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100)
    embedded_doc_set = doc_set.embed(embedder=embedder)  # OpenAIEmbedder(model_batch_size=3))

    new_docs = embedded_doc_set.take()

    assert len(new_docs) == len(docs)

    for doc in new_docs:
        assert len(doc.embedding) == 1536
