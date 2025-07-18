import argparse
from opensearchpy import OpenSearch
import numpy as np
import os
import pickle
import random
from rich.console import Console
from typing import List, Dict
import time
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from data_utils import extract_match_nomatch
import sycamore
from sycamore.connectors.opensearch.utils import get_knn_query, get_knn_query_vector
from sycamore.data import Document, Element
from sycamore.docset import DocSet

# from sycamore.functions import HuggingFaceTokenizer
from sycamore.query.client import SycamoreQueryClient
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.execution.sycamore_operator import QueryDatabase, QueryVectorDatabase, LlmFilter
from sycamore.llms import OpenAI, OpenAIModels
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.utils.opensearch import guess_opensearch_host


def get_docs(result, pickle_file=None):
    console = Console()
    if isinstance(result, DocSet):
        print("Got a docset back, forcing execution")
        result = result.take_all()
        console.rule("Docset query result")
        print(f"Got {len(result)} documents back")
        # print(result[:1])
        if pickle_file is not None:
            with open(pickle_file, "wb") as f:
                pickle.dump(result, f)
        return [doc["doc_id"] for doc in result]
    else:
        console.rule("Non-docset query result")
        print("Got a non-docset back")
        print(result)


def make_element_from_text(text: str) -> Element:

    return Element(text_representation=text)


#    return {'properties': {}, 'text_representation': text}


def generate_elements(keyword_to_probability: Dict[str, float], max_elements: int) -> list:

    strings = []
    num = random.randint(1, max_elements)
    base_text = " Lorem ipsum dolor sit amet. "
    for i in range(num):
        text = base_text
        # choose keywords to include based on the probability of each keyword
        selected_keywords = " ".join([k for k in keyword_to_probability if random.random() < keyword_to_probability[k]])
        print(f"Selected keywords: {selected_keywords}")
        strings.append(make_element_from_text(text + selected_keywords))
    return strings


def generate_doc(doc_id: str, keyword_to_probability: Dict[str, float], maxelems: int) -> Document:
    """
    Generate a document with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        maxelems (int): Maximum number of elements in each document.

    Returns:
        A Document objects with a random set of elements.
    """
    return Document(doc_id=doc_id, elements=generate_elements(keyword_to_probability, maxelems))


#    return {'doc_id':doc_id, 'elements':generate_elements(keywords, probabilities, maxelems)}


def generate_docset(keyword_to_probability: Dict[str, float], numdocs: int, maxelems: int) -> List[Document]:
    """
    Generate a document set with random elements.

    Args:
        keywords (list): List of keywords.
        probabilities (list): List of probabilities corresponding to the keywords.
        numdocs (int): Maximum number of documents in the docset.
        maxelems (int): Maximum number of elements in each document.

    Returns:
        list: A list of Document objects, each containing a random set of elements.
    """
    docset = []
    for i in range(numdocs):
        doc = generate_doc(str(i), keyword_to_probability, maxelems)
        docset.append(doc)
    return docset


def create_search_plan(index, query, keyword):
    return LogicalPlan(
        query=keyword,
        result_node=0,
        nodes={
            0: QueryDatabase(
                node_type="QueryDatabase",
                node_id=0,
                description="Get all documents",
                query={"bool": {"must": [{"match_phrase": {"text_representation": query}}]}},
                inputs=[],
                index=index,
            ),
        },
    )


def create_qvdb_plan(index, query, keyword):
    return LogicalPlan(
        query=keyword,
        result_node=0,
        nodes={
            0: QueryVectorDatabase(
                description="Get docs involving " + query,
                index=index,
                query_phrase=keyword,
                node_id=0,
            ),
        },
    )


def create_llm_filter_plan(index, query):
    return LogicalPlan(
        query=query,
        result_node=1,
        nodes={
            0: QueryDatabase(
                node_type="QueryDatabase",
                node_id=0,
                description="Get all documents",
                query={"match_all": {}},
                inputs=[],
                index=index,
            ),
            1: LlmFilter(
                description="Filter documents involving " + query,
                field="text_representation",
                question=query,
                node_id=1,
                inputs=[0],
            ),
        },
    )


def create_get_all_plan(index):
    return LogicalPlan(
        query="all",
        result_node=0,
        nodes={
            0: QueryDatabase(
                node_type="QueryDatabase",
                node_id=0,
                description="Get all documents",
                query={"match_all": {}},
                inputs=[],
                index=index,
            ),
        },
    )


def create_knn_plan(index, query, context, embedder):
    return LogicalPlan(
        query=query,
        result_node=0,
        nodes={
            0: QueryVectorDatabase(
                node_type="QueryVectorDatabase",
                node_id=0,
                description="Get all documents",
                # query=get_knn_query(query_phrase=keyword, k=1000, context=context, text_embedder=embedder),
                inputs=[],
                query_phrase=query,
                index=index,
                k=1000,
            ),
        },
    )


def prune_after_score_drop(rs, frac):
    if len(rs) == 0:
        return rs

    last_score = rs[0][1]
    for i in range(1, len(rs)):
        if rs[i][1] < last_score * frac:
            return rs[0:i]
        last_score = rs[i][1]

    return rs


def query(lp, context, pickle_file=None):
    start_time = time.time()
    # Existing code for the query function
    llm = OpenAI(OpenAIModels.GPT_4O.value)
    client = SycamoreQueryClient(llm=llm)
    result = get_docs(client.run_plan(lp).result, pickle_file)
    #    print(f"query generated {len(result)} results")
    #    print(result)
    # Existing code for the query function
    end_time = time.time()
    execution_time = end_time - start_time
    return (result, execution_time)


def query_keyword(os_client, keyword, index):
    query = {"query": {"match": {"text_representation": keyword}}, "size": 10000}

    response = os_client.search(index=index, body=query)

    chunks = [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]]
    docs = [hit["_source"]["parent_id"] for hit in response["hits"]["hits"]]
    return (len(chunks), len(set(docs)))


def query_knn(keyword, index, context, text_embedder, os_client_args):
    start_time = time.time()
    os_query = get_knn_query(query_phrase=keyword, context=context, k=10000, text_embedder=text_embedder)
    print(f"os_query: {os_query}")
    docset = context.read.opensearch(
        index_name=index, query=os_query, os_client_args=os_client_args, reconstruct_document=True
    )
    result = get_docs(docset)
    end_time = time.time()
    execution_time = end_time - start_time
    return (result, execution_time)


def query_knn_vector(vector, index, context, text_embedder, os_client_args, k=10000):
    start_time = time.time()
    os_query = get_knn_query_vector(query_vector=vector, context=context, k=k)
    # print(f'os_query: {os_query}')
    docset = context.read.opensearch(
        index_name=index, query=os_query, os_client_args=os_client_args, reconstruct_document=True
    )
    result = get_docs(docset)
    end_time = time.time()
    execution_time = end_time - start_time
    return (result, execution_time)


def chunked_sums(arr, N):
    return [sum(arr[0 : i + N]) for i in range(0, len(arr), N)]


def zero_run_lengths(arr):
    result = []
    count = 0
    for val in arr:
        if val == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result


def get_doc_data(data, doc_id):
    """
    Get the data for a specific document ID from the list of documents.
    """
    print(f"Getting data for doc_id: {doc_id}")
    for doc in data:
        print("Checking ")
        print(doc["doc_id"])
        if doc["doc_id"] == doc_id:
            return doc
    return None


def sort_and_score_knn_direct(
    lf_result, keyword, index, os_client, os_client_args, context, k, text_embedder, inverse=False
):
    bucket_size = 5
    console = Console()
    console.rule("1. KNN Sort")
    start_time = time.time()
    os_query = get_knn_query(query_phrase=keyword, context=context, k=k, text_embedder=text_embedder)
    os_query["size"] = k
    # print(f'KNN os_query: {os_query}')
    # Search the index
    response = os_client.search(index=index, body=os_query)
    all_docs = []
    for hit in response["hits"]["hits"]:
        if hit["_source"]["parent_id"] not in all_docs:
            all_docs.append(hit["_source"]["parent_id"])
    [(hit["_source"]["parent_id"], hit["_score"]) for hit in response["hits"]["hits"]]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"All docs based on knn_query: {len(all_docs)} deduped {len(set(all_docs))}")
    print(f"Pickled results: {len(lf_result)}")

    # Filter lf_results to only docs that are in the KNN results
    lf_result_filtered = [doc for doc in lf_result if doc["doc_id"] in all_docs]
    (e_match, e_nomatch, no_batches, elems_processed) = extract_match_nomatch(lf_result_filtered)

    lf_doc_ids = [doc["doc_id"] for doc in lf_result]
    results = [1 if d in lf_doc_ids else 0 for d in all_docs]

    sums = chunked_sums(results, bucket_size)
    zrl = zero_run_lengths(results)

    console.rule("2. Clustering")
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit_predict(e_match)
    centroids = kmeans.cluster_centers_
    docs_like_centroids_scores = []
    for c in centroids:
        os_vect_query = get_knn_query_vector(query_vector=c, k=k)
        os_vect_query["size"] = k
        # print(f'KNN os_query: {os_query}')
        # Search the index
        response = os_client.search(index=index, body=os_vect_query)
        docs_like_c = []
        docs_like_c_vects = []

        for hit in response["hits"]["hits"]:
            if hit["_source"]["parent_id"] not in docs_like_c:
                docs_like_c.append(hit["_source"]["parent_id"])
                docs_like_c_vects.append([hit["_source"]["parent_id"], hit["_source"]["embedding"]])
                docs_like_c_scores = [(hit["_source"]["parent_id"], hit["_score"]) for hit in response["hits"]["hits"]]
        # print(f'Docs like c: {docs_like_c[0]}')
        docs_like_centroids_scores.extend(docs_like_c_scores)
    sorted_docs_like_centroids_scores = sorted(docs_like_centroids_scores, key=lambda x: x[1], reverse=True)
    docs_like_centroids_deduped = []
    for doc_id, score in sorted_docs_like_centroids_scores:
        if doc_id not in docs_like_centroids_deduped:
            docs_like_centroids_deduped.append(doc_id)

    prior_docs = set(all_docs)
    new_docs = set(docs_like_centroids_deduped).difference(prior_docs)
    print(f"Got {len(new_docs)} new docs")

    new_docs_sorted = [d for d in docs_like_centroids_deduped if d in new_docs]
    # console.rule("3. Classify")
    # # Combine and create labels
    # X = np.array(e_match + e_nomatch)
    # y = np.array([1] * len(e_match) + [0] * len(e_nomatch))
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(X, y)

    # # filtered_docs = [d for d in new_docs if clf.predict([get_matching_elem_vect(get_doc_data(lf_result, d))])[0] == 1]
    # filtered_docs_1 = [d for d in docs_like_c_vects if clf.predict(np.array(d[1]).reshape(1, -1))[0] == 1]
    # filtered_docs_0 = [d for d in docs_like_c_vects if clf.predict(np.array(d[1]).reshape(1, -1))[0] == 0]
    # print(f'Filtered docs 1: {len(filtered_docs_1)}, Filtered docs 0: {len(filtered_docs_0)}')

    # think a bit more about this - don't count twice teh docs already processed
    more_results = [1 if d in lf_doc_ids else 0 for d in new_docs_sorted]
    results.extend(more_results)

    print(f"Docs after centroid based retrieval results {len(results)}")

    more_sums = chunked_sums(results, bucket_size)
    more_zrl = zero_run_lengths(results)
    # print(f'Results: {results}')

    return (sums, zrl, more_sums, more_zrl, execution_time)


def query_knn_direct(keyword, index, os_client, context, k, text_embedder, inverse=False):
    console = Console()
    console.rule()
    console.rule("KNN Query")
    start_time = time.time()
    os_query = get_knn_query(query_phrase=keyword, context=context, k=k, text_embedder=text_embedder)
    os_query["size"] = k
    # print(f'KNN os_query: {os_query}')
    if inverse:
        embedding_np = np.array(os_query["query"]["knn"]["embedding"]["vector"])
        inverted = 1 - embedding_np
        os_query["query"]["knn"]["embedding"]["vector"] = inverted.tolist()
        print(f"KNN os_query: {os_query}")
    # Search the index
    response = os_client.search(index=index, body=os_query)

    # Print results
    all_results = [hit for hit in response["hits"]["hits"]]
    all_with_keyword = [r for r in all_results if keyword in r["_source"]["text_representation"].split(" ")]
    all_without_keyword = [r for r in all_results if keyword not in r["_source"]["text_representation"].split(" ")]
    print(f"Got {len(all_results)} hits")
    print(f"With keyword: {len(all_with_keyword)}")
    if len(all_with_keyword) > 0:
        print(all_with_keyword[0])
    print(f"Without keyword: {len(all_without_keyword)}")
    if len(all_without_keyword) > 0:
        print(all_without_keyword[0])
    #   embedding1 = all_with_keyword[0]['_source']['embedding']
    #   embedding2 = all_without_keyword[0]['_source']['embedding']
    #   similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    #   print(f"Cosine similarity: {similarity:.4f}")

    # Difference vector
    # difference_vector = embedding1 - embedding2
    # print(f"Difference vector (first 5 values): {difference_vector[:5]}")

    all_docs_scores = [(hit["_source"]["parent_id"], hit["_score"]) for hit in response["hits"]["hits"]]

    all_results_sorted = sorted(all_docs_scores, key=lambda x: x[1], reverse=True)
    pruned_results = prune_after_score_drop(all_results_sorted, 0.001)
    all_doc_ids = [doc_id for doc_id, score in all_results_sorted]
    print(f"KNN Results: count {len(all_results_sorted)}, First {all_results_sorted[0]}, last {all_results_sorted[-1]}")
    print(f"KNN Pruned: count {len(pruned_results)}, First {pruned_results[0]}, last{pruned_results[-1]}")
    print(f"KNN Docs {len(all_doc_ids)}")
    print(f"KNN Dedupped {len(set(all_doc_ids))}")
    print(f"Max : {max(all_results_sorted, key=lambda x: x[1])}")
    print(f"Min : {min(all_results_sorted, key=lambda x: x[1])}")
    result = [doc_id for doc_id, score in pruned_results]
    end_time = time.time()
    execution_time = end_time - start_time
    return (result, execution_time)


def print_index_stats(INDEX, os_client, keywords_to_probability):

    chunks = os_client.count(index=INDEX)
    print("Element count:", chunks["count"])
    for k, p in keywords_to_probability.items():
        print(f"{k} elements/docs: {query_keyword(os_client, k, INDEX)}")


def main():
    argparser = argparse.ArgumentParser(prog="synth_data_llm_filter")
    argparser.add_argument(
        "--oshost", default=None, help="OpenSearch host. Defaults to guessing based on whether it is in a container."
    )
    argparser.add_argument("--osport", default=9200, help="OpenSearch port to use, Defaults to 9200")
    argparser.add_argument("--numdocs", default=10, help="Number of documents to generate")
    argparser.add_argument("--maxelems", default=5, help="Maximum number of elements in each document")
    argparser.add_argument("--index", default=None, help="The OpenSearch index name to populate")
    argparser.add_argument("--query", default=None, help="Perform a query")
    argparser.add_argument("--query_rank", default=None, help="Perform a query")
    argparser.add_argument("--k", default=None, help="k in KNN")
    argparser.add_argument("--stats", action="store_true")
    argparser.add_argument("--pickleout", default=None, help="If present, output is dumped there")
    argparser.add_argument("--picklein", default=None, help="If present, input is read from there")

    args = argparser.parse_args()

    # The OpenSearch index name to populate.
    if args.index is not None:
        INDEX = args.index
    else:
        INDEX = "synth_10"

    if args.oshost is not None:
        opensearch_host = args.oshost
    else:
        opensearch_host = guess_opensearch_host()

    opensearch_port = args.osport

    os_client_args = {
        "hosts": [{"host": opensearch_host, "port": opensearch_port}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": True,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }

    os_client = OpenSearch(**os_client_args)

    index_settings = {
        "body": {
            "settings": {"index.knn": True, "number_of_shards": 5, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "embedding": {
                        "dimension": 384,
                        "method": {
                            "engine": "faiss",
                            "space_type": "l2",
                            "name": "hnsw",
                            "parameters": {},
                        },
                        "type": "knn_vector",
                    }
                }
            },
        }
    }

    embedder = SentenceTransformerEmbedder(batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2")
    # The number of documents to generate.
    numdocs = int(args.numdocs)

    # The maximum number of elements in each document.
    maxelems = int(args.maxelems)

    keyword_to_probability = {
        "banana": 0.015,
        "cat": 0.01,
        "dog": 0.02,
    }

    console = Console()
    console.rule("Using index " + INDEX)

    if args.stats:
        print_index_stats(INDEX, os_client, keyword_to_probability)
        return

    if args.query:
        context = sycamore.init()
        keyword = args.query
        k = int(args.k)

        lp_lf = create_llm_filter_plan(INDEX, keyword)
        if args.pickleout is not None:
            pickle_file = args.pickleout
            if os.path.exists(os.path.dirname(pickle_file)):
                rs_lf, time_lf = query(lp_lf, context, pickle_file=pickle_file)
            else:
                rs_lf, time_lf = query(lp_lf, context)
        print(f"Llm_Filter Query returned {len(rs_lf)} results in {time_lf} seconds")

        return

    if args.query_rank:
        context = sycamore.init()
        keyword = args.query_rank
        k = int(args.k)
        # lp_lf = create_llm_filter_plan(INDEX, keyword)

        # Read stored llm_filter results
        if args.picklein is None:
            print("Need to provide a pickle file with the results of the llm_filter")
            return

        # Load the pickle file
        with open(args.picklein, "rb") as f:
            data = pickle.load(f)  # should be a list of dicts
        doc_ids = [data["doc_id"] for data in data]
        print(
            f"Loading Llm_Filter Query resutls from {args.picklein}\n - Loaded total \n {len(doc_ids)}, disitinct {len(set(doc_ids))} results"
        )

        (chunked_sums, zero_subseq, chunked_sums_after_cluster, zero_subseq_after_cluster, time) = (
            sort_and_score_knn_direct(data, keyword, INDEX, os_client, os_client_args, context, k, embedder)
        )
        print(f"Sort and score KNN returned:\n {chunked_sums} in {time} seconds")
        print(f"Sort and score KNN after cluster:\n {chunked_sums_after_cluster}\n in {time} seconds")
        return

    docset = generate_docset(keyword_to_probability, numdocs, maxelems)

    print(f"Generated docset containing {len(docset)} documents")
    print(docset[0])

    context = sycamore.init()
    # tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
    # llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)

    context.read.document(docs=docset).explode().embed(embedder=embedder).write.opensearch(
        os_client_args=os_client_args, index_name=INDEX, index_settings=index_settings
    )

    print_index_stats(INDEX, os_client, keyword_to_probability)


if __name__ == "__main__":
    exit(main())
