import json
import os
import time
from pathlib import Path
from typing import Any

import datasets

import sycamore
from sycamore.connectors.file.file_writer import JSONEncodeWithUserDict
from sycamore.data import Element
from sycamore.evaluation import EvaluationDataPoint
from sycamore.evaluation.metrics import document_retrieval_metrics, rouge_metrics
from sycamore.evaluation.datasets import EvaluationDataSetReader
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.transforms.embed import SentenceTransformerEmbedder
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.evaluation.subtasks import SubtaskExecutor

subtask_path = "/home/admin/subtask_info_copy.json"

with open(subtask_path) as json_file:
    data = json.load(json_file)

calcs_reqd = data["calculation_ids"]

no_year_ids = [
    "financebench_id_01858",
    "financebench_id_07966",
    "financebench_id_07507",
    "financebench_id_08135",
    "financebench_id_00799",
    "financebench_id_01079",
    "financebench_id_01148",
    "financebench_id_01930",
    "financebench_id_00563",
    "financebench_id_01351",
    "financebench_id_02608",
    "financebench_id_01077",
    "financebench_id_00288",
    "financebench_id_00460",
    "financebench_id_03838",
    "financebench_id_00464",
    "financebench_id_00585",
    "financebench_id_02981",
    "financebench_id_01346",
    "financebench_id_01107",
    "financebench_id_00839",
    "financebench_id_03718",
    "financebench_id_03849",
    "financebench_id_00552",
    "financebench_id_04302",
    "financebench_id_00735",
    "financebench_id_00302",
    "financebench_id_00283",
    "financebench_id_00521",
    "financebench_id_00605",
    "financebench_id_00566",
    "financebench_id_04784",
    "financebench_id_06741",
]

import re
import pandas as pd


def extract_year(question, company):
    pattern = r"\bFY\d{2}\b|\b\d{4}\b|\bFY\d{4}\b"
    yrs = re.findall(pattern, question)

    yrs = [yr[-2:] for yr in yrs]

    year = ("20" + max(yrs)) if len(yrs) != 0 else ""

    return doc_exists(year, company)


def doc_exists(year, company):
    df = pd.read_csv("/home/admin/financebench_sample_150.csv")
    df = df["doc_name"].str.split("_", expand=True)

    df = df.loc[(df[0] == company) & (df[1].str.startswith(year))]

    return "" if df.empty else year


def _hf_to_qa_datapoint(datapoint: dict[str, Any]) -> dict[str, Any]:
    document = EvaluationDataPoint()

    page_numbers = [int(num.strip()) for num in datapoint["page_number"].split(",")]

    document.question = datapoint["question"]
    document.ground_truth_answer = datapoint["answer"]
    document.ground_truth_source_documents = [
        Element(
            {
                "properties": {
                    "_location": datapoint["doc_link"],
                    "page_number": page_numbers[0],
                    "page_numbers": page_numbers,
                }
            }
        )
    ]

    document.doc_id = datapoint["financebench_id"]
    document.properties["subtasks_reqd"] = True if document.doc_id in calcs_reqd else False

    # filters
    company = datapoint["doc_name"].split("_")[0]
    year = datapoint["doc_period"] if document.doc_id not in no_year_ids else extract_year(document.question, company)
    document.filters = {"properties.company": company}
    if year:
        document.filters["properties.year"] = int(year)
    # print ("QN", document["question"])
    if document.properties["subtasks_reqd"]:
        document.properties["subtask_filters"] = {
            "company": company,
            "year": int(year),
            "year1": int(year) - 1,
            "year2": int(year) - 2,
        }

    document["raw"] = datapoint
    return {"doc": document.serialize()}


INDEX = "textract-mpnet"

if os.path.exists("/.dockerenv"):
    opensearch_host = "opensearch"
    print("Assuming we are in a sycamore jupyter container, using opensearch for opensearch host")
else:
    opensearch_host = "localhost"
    print("Assuming we are running outside of a container, using localhost for opensearch host")

OS_CLIENT_ARGS = {
    "hosts": [{"host": opensearch_host, "port": 9200}],
    "http_compress": True,
    "http_auth": ("admin", "admin"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "timeout": 120,
}

OS_CONFIG = {
    "size": 10,
    "neural_search_k": 200,
    "embedding_model_id": "hlAX5Y8BnK-z0ftijBv_",
    "search_pipeline": "hybrid_rag_pipeline",
    "llm": "gpt-4o",
    "context_window": "10",
}

base_path = str(Path(__file__).parent)
output_path = "new-knn-subtasks.json"
output_path2 = "nosub-knn-4o.json"

context = sycamore.init()
reader = EvaluationDataSetReader(context)
hf_dataset = datasets.load_dataset("PatronusAI/financebench", split="train[0:150]")
input_docset = reader.huggingface(hf_dataset, doc_extractor=_hf_to_qa_datapoint)
embedder = SentenceTransformerEmbedder(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=100)

sub_exec = SubtaskExecutor(
    subtask_path,
    None,
    INDEX,
    OS_CONFIG,
    query_executor=OpenSearchQueryExecutor(OS_CLIENT_ARGS),
    embedder=embedder,
    knn_query=True,
)

start = time.time()
subtask_docs = sub_exec.execute(input_docset)


data = {
    "experiment_name": "FinanceBench gpt-4o textract knn mpnet",
    "description": "gpt-4o",
    "created_by": "aanyapratapneni",
    "index": INDEX,
    "os_client_args": OS_CLIENT_ARGS,
    "os_config": OS_CONFIG,
    "qa_path": ["huggingface: PatronusAI/financebench"],
}

pipeline = EvaluationPipeline(
    index=INDEX,
    os_config=OS_CONFIG,
    metrics=[document_retrieval_metrics, rouge_metrics],
    query_executor=OpenSearchQueryExecutor(OS_CLIENT_ARGS),
    subtask_docs=subtask_docs,
    embedder=embedder,
    knn_query=True,
)

start = time.time()
query_level_metrics = pipeline.execute(input_docset)[0]
data["query_level_data"] = query_level_metrics.take_all()
data["evaluation_time"] = f'{"{:.2f}".format(time.time() - start)} seconds'
print("COMPLETE")
with open(output_path2, "w+") as outfile:
    json.dump(data, outfile, cls=JSONEncodeWithUserDict)
    print("FINAL_DUMPED")
