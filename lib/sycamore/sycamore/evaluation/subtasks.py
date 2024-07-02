import json
from typing import Any
import sycamore
from sycamore.data.document import Document
from sycamore.docset import DocSet
from sycamore.evaluation.data import EvaluationDataPoint
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import TaskIdentifierZeroShotGuidancePrompt
from sycamore.scans.materialized_scan import DocScan
from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.writers.file_writer import JSONEncodeWithUserDict


task_list = {}
task_descriptions = ""
openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
prompt = TaskIdentifierZeroShotGuidancePrompt() # TODO@aanya: create a new prompt with question, tasks

INDEX = "metadata-eval-2.0-mpnet"

OS_CLIENT_ARGS = {
    "hosts": [{"host": "localhost", "port": 9200}],
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
    "llm": "gpt-3.5-turbo",
    "context_window": "10",
}

task_descriptions = {
    "CAPINT": "Questions that inquire whether a company is capital intensive",
    "QR": "Questions that ask about the quick ratio of a company",
    "FATR": "Questions that inquire about the fixed asset turnover ratio of a company",
    "CAPREV3": "Questions that ask about the 3 year average of capex as a percentage of revenue",
    "ITR": "Questions about inventory turnover ratio"
}

task_list = {
    "CAPINT": [
        "What is the capital expenditure of {company} in {year}? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "What is the revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Divide the capital expenditure by revenue. If this ratio is less than 1, {company} is not capital intensive. If the ratio is greater than 1, {company} is capital intensive. Calculate this ratio and determine whether {company} is capital intensive."
               ],
    "QR": [
        "What is the total current assets of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "What is the total current liabilities of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "Find the quick ratio using the formula (assets – inventory)/liabilities. If the quick ratio is above 1, {company} has a healthy liquidity profile. Else, it does not. Use this metric to answer the following question."
    ],
    "FATR": [
        "What is the total revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "What is the capital expenditure of {company} in {year}? What is the capital expenditure of {company} in the previous year? These values can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "Calculate the fixed asset turnover ratio using the formula (2*revenue / (capital expenditure in {year} + capital expenditure in previous year)). Use this metric to answer the following question."
    ],
    "CAPREV3": [
        "What is the capital expenditure of {company} in {year}, in the previous year, and 2 years prior? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "What is the revenue of {company} in {year}, in the previous year, and 2 years prior? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Find the average capital expenditure across all three years. Find the average revenue across all three years. Divide average expenditure by average revenue. Use this metric to answer the following question. Show the calculation in your answer."
    ],
    "ITR": [
        "What is the cost of sales (also referred to as cost of goods sold) of {company} in {year}? This value can be found on the consolidated statements of operations.",
        "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "The inventory turnover ratio is the cost of sales or cost of goods sold divided by inventory. Calculate this metric, then use it to answer the following question."
    ]
}

def subtask_to_qa_datapoint(question: str, company: str, year: str) -> dict[str, Any]:
    document = EvaluationDataPoint()
    document.question = question
    document.additional_info = {"subtask": True, "company": company, "year": year}

    document["raw"] = question
    return document


def collector(subtasks, company, year):
    # called by executor
    # set up input docset containing questions from subtask list
        # need to figure out how to set this up: likely need a datapoint to docset converter (_hf_to_qa_datapoint type)
        # call question to eval data point function on all subtasks
        # get list of eval data points
        # give this to DocScan, return json_scan
        # create docset with json_scan, context
    # create EvaluationPipeline object
    # modify EvaluationPipeline execute to not aggregate
    # collect all generated answers, return

    docs = [subtask_to_qa_datapoint(subtask, company, year) for subtask in subtasks]
    json_scan = DocScan(docs=docs)
    
    context = sycamore.init()
    input_docset = DocSet(context, json_scan)

    pipeline = EvaluationPipeline(
        index=INDEX,
        os_config=OS_CONFIG,
        query_executor=OpenSearchQueryExecutor(OS_CLIENT_ARGS),
        metrics=[]
    )

    query_level_metrics = pipeline.subtask_execute(input_docset)
    results = query_level_metrics.take_all()
    answers = [result["generated_answer"] for result in results]

    return answers


def executor(question, company, year):
    # calls openai to find task id
    # uses knowledge document
    # calls collector on list of subtasks
    # append query results to final query, return
    task_id = openai_llm.generate(
        prompt_kwargs={
            "prompt": prompt,
            "question": question,
            "task_descriptions": task_descriptions,
        }
    )

    print (task_id)

    subtask_list = task_list[task_id]
    results = collector(subtask_list[:-1], company, year)
    print (results)

    return " ".join(results) + subtask_list[-1] if results else "Failed"