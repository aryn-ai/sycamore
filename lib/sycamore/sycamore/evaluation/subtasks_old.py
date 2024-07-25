import json
from typing import Any
import sycamore
from sycamore.connectors.file.materialized_scan import DocScan
from sycamore.data.document import Document
from sycamore.docset import DocSet
from sycamore.evaluation.data import EvaluationDataPoint
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import TaskIdentifierZeroShotGuidancePrompt
from sycamore.transforms.query import OpenSearchQueryExecutor


task_list = {}
task_descriptions = ""
openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
prompt = TaskIdentifierZeroShotGuidancePrompt()

INDEX = "textract-mpnet"

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
    "ITR": "Questions about inventory turnover ratio",
    "ROA": "Questions about return on assets (ROA)",
    "DPO": "Questions about days payable outstanding",
    "GM2": "Questions about gross margin profile relative to the previous year",
    "EBITDA": "Questions about EBITDA",
    "WC": "Questions about a company's working capital",
    "ANPM3": "Questions about the 3 year average net profit margin of a company",
    "WCR": "Questions about a company's working capital ratio",
    "COGSM": "Questions about a company's COGS percentage margin",
    "DPR": "Questions about dividend payout ratio",
    "AUOIM3": "Questions about 3 year average unadjusted operating income percentage margin",
    "CCC": "Questions about cash conversion cycle",
    "FCF": "Questions about free cash flow",
    "RR": "Questions about retention ratio",
    "SA": "Questions about what shareholders receive if a company liquidates all assets",
    "ATR": "Questions about asset turnover ratio",
    "CAGR2": "Questions about 2 year total revenue CAGR",
    "EBITDAM": "Questions about the ebitda percentage margin of a company",
    "COGSREV3": "Questions about the 3 year average of cost of goods sold as a percentage of revenue",
    "CAPEX": "Questions about only the capital expenditure of a company and no additional information or metrics",
    "ETR2": "Questions about the effective tax rate of a company relative to the previous year"
}

task_descriptions_f = {
    "CAPINT": "Questions that inquire whether a company is capital intensive",
    "WC": "Questions about a company's working capital",
}

task_formulae = {
    "CAPINT": ["CAPEX/REV", "If this ratio is less than 1, {company} is not capital intensive. If the ratio is greater than 1, {company} is capital intensive."],
    "WC": ["TCA-TCL"]
}

metric_qs = {
    "TCA": "What is the total current assets of {company} in {year}? This value can be found on the balance sheet.",
    "TCL": "What is the total current liabilities of {company} in {year}? This value can be found on the balance sheet.",
}

task_list = {
    # "CAPINT": [
    #     # "What is the capital expenditure of {company} in {year}? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
    #     # "What is the revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
    #     "Divide the capital expenditure of {company} by its revenue. If this ratio is less than 1, {company} is not capital intensive. If the ratio is greater than 1, {company} is capital intensive."
    # ],
    "CAPINT": [
        "What is the capital expenditure of {company} in {year}? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "What is the revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Divide the capital expenditure of {company} by its revenue. If this ratio is less than 1, {company} is not capital intensive. If the ratio is greater than 1, {company} is capital intensive. Calculate this ratio and determine whether {company} is capital intensive."
    ],
    # "QR": [
    #     # "What is the total current assets of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     # "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     # "What is the total current liabilities of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     # "Find the phrase 'total current liabilities' in the consolidated balance sheet. What is the first value listed beside it?",
    #     "Find the quick ratio using the formula (total current assets – total inventories)/total current liabilities. If the quick ratio is above 1, {company} has a healthy liquidity profile. Else, it does not. Use this metric to answer the following question."
    # ],
    "QR": [
        "What is the total current assets of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet.",
        "What is the prepaid expenses (as part of current assets) of {company} in {year}? This value can be found on the consolidated balance sheet. If it cannot be found, return prepaid expenses as 0.",
        "What is the total current liabilities of {company} in {year}? This value can be found on the consolidated balance sheet.",
        # "Find the phrase 'total current liabilities' in the consolidated balance sheet. What is the first value listed beside it?",
        "Find the quick ratio using the formula (total current assets – total inventories - prepaid expenses)/total current liabilities. If the quick ratio is above 1, {company} has a healthy liquidity profile. Else, it does not. Use this metric to answer the following question."
    ],
    # "QR": [
    #     "What is the cash and cash equivalents of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     "What is the short term investments of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     "What is the accounts receivable, net of {company} in {year}? This value can be found on the consolidated balance sheet. If it cannot be found, return prepaid expenses as 0.",
    #     "What is the receivables from related parties of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     "What is the total current liabilities of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     # "Find the phrase 'total current liabilities' in the consolidated balance sheet. What is the first value listed beside it?",
    #     "Find the quick ratio using the formula (cash and cash equivalents + short term investments + accounts receivable, net + receivables from related parties)/total current liabilities. If the quick ratio is above 1, {company} has a healthy liquidity profile. Else, it does not. Use this metric to answer the following question."
    # ],
    "FATR": [
        "What is the total revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "What is the PP&E net of {company} in {year}? What is the PP&E net of {company} in the previous year? These values can be found in the balance sheet. PP&E stands for property, plant, and equipment.",
        "Calculate the fixed asset turnover ratio using the formula (2*revenue / (PP&E in {year} + capital expenditure in previous year)). Use this metric to answer the following question."
    ],
    # "CAPREV3": [
    #     # "What is the capital expenditure of {company} in {year}, in the previous year, and 2 years prior? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
    #     # "What is the revenue of {company} in {year}, in the previous year, and 2 years prior? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
    #     "Find the average capital expenditure of {company} in {year}, in the previous year, and 2 years prior. Find the revenue of {company} in {year}, in the previous year, and 2 years prior. Find the average expenditure across all three years and divide by average revenue across all three years. Use this metric to answer the following question."
    # ],
    "CAPREV3": [
        "What is the capital expenditure of {company} in {year}, in the previous year, and 2 years prior? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "What is the revenue of {company} in {year}, in the previous year, and 2 years prior? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Find the average expenditure across all three years and divide by average revenue across all three years. Use this metric to answer the following question."
    ],
    # "ITR": [
    #     # "What is the cost of sales (also referred to as cost of goods sold) of {company} in {year}? This value can be found on the consolidated statements of operations. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
    #     # "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet.",
    #     "Divide the cost of sales of {company} by its total inventory. Use this metric to answer the following question."
    # ],
    "ITR": [
        "What is the cost of sales (also referred to as cost of goods sold) of {company} in {year}? This value can be found on the consolidated statements of operations. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "What is the total inventories of {company} in {year}? This value can be found on the consolidated balance sheet. 'Total inventory' is also referred to as simply 'inventory'",
        "Divide the cost of sales of {company} by its total inventory. Use this metric to answer the following question."
    ],
    "ROA": [
        "What is the net income of {company} in {year}? This value can be found on the consolidated statement of cash flows.",
        "What is the total assets of {company} in {year}? This value can be found on the balance sheet.",
        "Calculate the return on assets (ROA) using the formula net income / total assets. Use this to answer the following question."
    ],
	"DPO": [
        "What is the cost of goods sold (COGS) for {company} in {year}? This value can be found on the income statement.  Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "What is the accounts payable for {company} in {year}? What is the accounts payable for {company} in the previous year? These values can be found on the balance sheet.",
        "What is the inventory for {company} in {year}? What is the inventory for {company} in the previous year? These values can be found on the balance sheet.",
        "Calculate the days payable outstanding (DPO) using the formula ((accounts payable in {year} + accounts payable in previous year) / (COGS in {year} + inventory in {year} - inventory in the previous year)) * 365. Use this to answer the following question."
    ],
    "GM2": [
        "What is the total revenue of {company} in {year}? What is the total revenue of {company} in the previous year? This value can be found on the income statement.",
        "What is the cost of goods sold (COGS) for {company} in {year}? What is the cost of goods sold (COGS) for {company} in the previous year? This value can be found on the income statement. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "Calculate the gross margin in {year} using the formula (revenue - COGS) / revenue. Calculate the gross margin in the previous year using the same formula. Use these values to answer the following question. Show all calculations."
    ],
    "EBITDA": [
        "What is the unadjusted operating income of {company} in {year}? This value can be found on the income statement.",
        "What is the depreciation and amortization amount for {company} in {year}? This value can be found on the cash flow statement.",
        "Calculate unadjusted operating income + depreciation and amortization to find unadjusted EBITDA. Use this metric to answer the following question."
    ],
    "WC": [
        "What is the total current assets of {company} in {year}? This value can be found on the balance sheet.",
        "What is the total current liabilities of {company} in {year}? This value can be found on the balance sheet.",
        "Calculate the working capital using the formula total current assets - total current liabilities. Use this metric to answer the following question."
    ],
    "ANPM3": [
        "What is the net income of {company} for {year}, the previous year, and the year before that? These values can be found on the income statements.",
        "What is the total revenue of {company} for {year}, the previous year, and the year before that? These values can be found on the income statements.",
        "Calculate the net profit margin for each year using the formula: net income / total revenue. Calculate the 3-year average net profit margin by averaging the net profit margins of the three years. Use this metric to answer the following question."
    ],
    "WCR": [
        "What is the total current assets of {company} in {year}? This value can be found on the balance sheet.",
        "What is the total current liabilities of {company} in {year}? This value can be found on the balance sheet.",
        "Calculate the working capital ratio using the formula total current assets / total current liabilities. Use this metric to answer the following question."
    ],
    "COGSM": [
        "What is the cost of goods sold (COGS) for {company} in {year}? This value can be found on the income statement. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "What is the net revenue of {company} in {year}? This value can be found on the income statement.",
        "Calculate the COGS percentage margin using the formula (COGS / net revenue) * 100. Use this metric to answer the following question."
    ],
    "DPR": [
        "What is the total dividends paid by {company} in {year}? This value can be found in the cash flow statement or the notes to the financial statements.",
        "What is the net income attributable to shareholders of {company} in {year}? This value can be found on the income statement.",
        "Calculate the dividend payout ratio using the formula (total dividends paid / net income attributable to shareholders) * 100. Use this metric to answer the following question."
    ],
    "AUOIM3": [
        "What is the unadjusted operating income of {company} for {year}, the previous year, and the year before that? This value can be found on the income statements or the cash flow statements.",
        "What is the total revenue of {company} for {year}, the previous year, and the year before that? This value can be found on the income statements.",
        "Calculate the unadjusted operating income percentage margin for each year using the formula (unadjusted operating income / total revenue) * 100. Calculate the 3-year average unadjusted operating income percentage margin by averaging the margins of the three years. This metric indicates the average profitability of {company} from its core operations over the specified period."
    ],
    "CCC": [
        "What is the inventory for {company} in {year}? What is the inventory for {company} in the previous year? These values can be found on the balance sheet.",
        "What is the cost of goods sold (COGS) for {company} in {year}? This value can be found on the income statement. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "What is the trade receivables for {company} in {year}? What is the trade receivables for {company} in the previous year? These values can be found on the balance sheet.",
        "What is the revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "What is the accounts payable for {company} in {year}? What is the accounts payable for {company} in the previous year? These values can be found on the balance sheet.",
        "Calculate DIO as 365 * ((inventory in {year} + inventory in previous year)/(2*COGS in {year})). Calculate DSO as 365 * ((trade receivables in {year} + trade receivables in previous year)/(2*revenue in {year})). Calculate DPO as ((accounts payable in {year} + accounts payable in previous year) / (COGS in {year} + inventory in {year} - inventory in the previous year)) * 182.5. Finally, calculate CCC as DIO + DSO - DPO. Use this metric to answer the following question."
    ],
    "FCF": [
        "What is the cash from operations for {company} in {year}? This value can be found on the cash flow statement.",
        "What is the capital expenditure of {company} in {year}? This value can be found in the cash flow statement. Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses.",
        "Calculate the free cash flow using the formula cash from operations - capital expenditure. Use this metric to answer the following question."
    ],
    "RR": [
        "What is the net income of {company} in {year}? This value can be found on the income statement.",
        "What is the total dividends paid by {company} in {year}? This value can be found in the cash flow statement or the notes to the financial statements.",
        "Calculate the retention ratio using the formula (net income - dividends paid) / net income. Use this metric to answer the following question."
    ],
    "SA": [
        "What is the TBVPS of {company} in {year}?",
        "The TBVPS is the amount that could be received by each stakeholder if {company} liquidated all its assets to pay its shareholders. Use this metric to answer the following question."
    ],
    "ATR": [
        "What is the total revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "What is the total assets of {company} in {year}? What is the total assets of {company} in the previous year? These values can be found in the balance sheet.",
        "Calculate the fixed asset turnover ratio using the formula (2*revenue / (total assets in {year} + total assets in previous year)). Use this metric to answer the following question."
    ],
    "CAGR2": [
        "What is the revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "What is the revenue of {company} in {year} - 2? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Calculate the 2-year total revenue CAGR as sqrt(revenue in {year}/revenue in {year} - 2) - 1. Use this metric to answer the following question."
    ],
    "EBITDAM": [
        "What is the unadjusted operating income of {company} in {year}? This value can be found on the income statement.",
        "What is the depreciation and amortization amount for {company} in {year}? This value can be found on the cash flow statement.",
        "What is the total revenue of {company} in {year}? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "Calculate EBITDA % margin using the formula: (unadjusted operating income + depreciation and amortization)*100/total revenue. Use this metric to answer the following question."
    ],
    "COGSREV3": [
        "What is the total cost of goods sold of {company} for {year}, the previous year, and the year before that? These values can be found on the income statements. Cost of goods sold is also referred to as 'total cost of revenue' or 'cost of sales'.",
        "What is the revenue of {company} in {year}, in the previous year, and 2 years prior? This value can be found on the first line of the income statement. Synonyms for revenue are: sales; net sales; net revenue.",
        "For each year, divide the cost of goods sold in that year by the revenue in that year. Find the average of this ratio across all three years. Use this metric to answer the following question. Show the calculation in your answer."
	],
    "CAPEX": [
        "Synonyms for capital expenditure are: capital spending; purchases of property, plant, and equipment (PP&E); acquisition expenses."
    ],
    "ETR2": [
        "What is the income tax expense/benefit of {company} in {year}? What is the income tax expense/benefit of {company} in the previous year? These values can be found on the consolidated statement of operations. If a value appears in parentheses, it is negative.",
        "What is the loss before income taxes of {company} in {year}? What is the loss before income taxes of {company} in the previous year? These values can be found on the consolidated statement of operations. If a value appears in parentheses, it is negative.",
        "For each year, divide the income tax expense/benefit by the loss before income taxes. This will output the effective tax rate for each year. Use this metric to answer the following question."
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

    if len(subtask_list) == 1:
        return subtask_list[0]
    
    results = collector(subtask_list[:-1], company, year)

    return " ".join(results) + subtask_list[-1] if results else ""