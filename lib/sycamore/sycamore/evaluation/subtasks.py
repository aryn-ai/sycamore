import json
from typing import Any
import sycamore
from sycamore.connectors.file.materialized_scan import DocScan
from sycamore.docset import DocSet
from sycamore.evaluation.data import EvaluationDataPoint
from sycamore.evaluation.pipeline import EvaluationPipeline
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import TaskIdentifierZeroShotGuidancePrompt
from sycamore.transforms.query import OpenSearchQueryExecutor

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
prompt = TaskIdentifierZeroShotGuidancePrompt()

def subtask_to_qa_datapoint(question: str, filters: dict[str, str], code: str) -> dict[str, Any]:
    question = question.format(**filters)

    document = EvaluationDataPoint()
    document.question = question + "Return only the code "+ code +" alongside the amount found and no additional information."
    document.filters = filters

    document["raw"] = question
    return document


def collector(terms, filters, instructions, index, query_executor, os_config):
    docs = [subtask_to_qa_datapoint(instructions[term], filters, term) for term in terms]
    json_scan = DocScan(docs=docs)
    
    context = sycamore.init()
    input_docset = DocSet(context, json_scan)

    pipeline = EvaluationPipeline(
        index=index,
        os_config=os_config,
        query_executor=query_executor,
        metrics=[],
        subtask=True
    )

    query_level_metrics = pipeline.execute(input_docset)[0]
    results = query_level_metrics.take_all()
    
    answers = []
    for result in results:
        answers.append(result["generated_answer"])
    
    return answers


def executor(question: str, filters: dict, filepath: str, index: str, query_executor: OpenSearchQueryExecutor, os_config: dict[str, str]):
    with open(filepath) as json_file:
        data = json.load(json_file)

    # LLM call to find formula
    task_id = openai_llm.generate(
        prompt_kwargs={
            "prompt": prompt,
            "question": question,
            "task_descriptions": data["task_descriptions"],
        }
    )

    task_formulas = data["task_formulas"]
    subtask_instructions = data["subtask_instructions"]

    # only instructions to be added to question: ["", instructions]
    if not task_formulas[task_id][0]:
        return task_formulas[task_id][1]

    final_q = "Use the given formula and dictionary of values to answer the question that follows. "

    # 1 formula to be processed: [formula, instructions]
    if len(task_formulas[task_id]) < 3:
        formula = task_formulas[task_id][0]
        terms = [term for term in subtask_instructions.keys() if term in formula]
        results = collector(terms, filters, subtask_instructions, index, query_executor, os_config)
        final_q = "Formula: " + task_id + "=" + formula + " Values: [" + ", ".join(results) + "]. "

    # multiple formulas to be processed: [formula 1, formula 2, ..., instructions]
    else:
        for formula in task_formulas[task_id][:-1]:
            terms = [term for term in subtask_instructions.keys() if term in formula]
            results = collector(terms, filters, subtask_instructions, index, query_executor, os_config)
            final_q += "Formula: " + formula + " Values: ["  + ", ".join(results) + "]. "

    # instructions provided
    if len(task_formulas[task_id]) > 1:
        final_q += task_formulas[task_id][-1].format(**filters)
    
    return final_q