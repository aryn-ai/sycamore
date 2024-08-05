import json
from typing import Any, Optional
from sycamore.data.document import Document, OpenSearchQuery
from sycamore.data.element import Element
from sycamore.docset import DocSet
from sycamore.llms.llms import LLM
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import SimpleGuidancePrompt, TaskIdentifierZeroShotGuidancePrompt
from sycamore.transforms.embed import Embedder, SentenceTransformerEmbedder
from sycamore.transforms.query import QueryExecutor


class SubtaskExecutor:
    def __init__(
        self,
        filepath: Optional[str],
        subtask_data: Optional[dict[str, Any]],
        index: str,
        os_config: dict[str, Any],
        query_executor: QueryExecutor,
        embedder: Embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=100
        ),
        llm: LLM = OpenAI(OpenAIModels.GPT_3_5_TURBO.value),
        prompt: SimpleGuidancePrompt = TaskIdentifierZeroShotGuidancePrompt(),
    ):
        if subtask_data:
            self._subtask_data = subtask_data
        elif filepath:
            with open(filepath) as json_file:
                self._subtask_data = json.load(json_file)
        else:
            raise RuntimeError("Need to provide a subtask data file or dictionary")

        self._index = index
        self._os_config = os_config
        self._query_executor = query_executor
        self._embedder = embedder
        self._llm = llm
        self._prompt = prompt

    def _get_formulas(self, document: Document) -> list[Document]:
        f_list = []
        if document.properties["subtasks_reqd"]:
            task_id = self._llm.generate(
                prompt_kwargs={
                    "prompt": self._prompt,
                    "question": document["question"],
                    "task_descriptions": self._subtask_data["task_descriptions"],
                }
            )

            formulas = self._subtask_data["task_formulas"][task_id].get("formulas", [])
            instructions = self._subtask_data["task_formulas"][task_id].get("instructions", "")

            for formula in formulas:
                doc = Document()
                doc.text_representation = formula
                doc.parent_id = document.doc_id
                doc.properties["instructions"] = instructions
                doc.properties["filters"] = document["filters"]
                doc.properties["subtask_filters"] = document.properties["subtask_filters"]
                f_list.append(doc)

            if not formulas:
                doc = Document()
                doc.text_representation = ""
                doc.parent_id = document.doc_id
                doc.properties["instructions"] = instructions
                f_list.append(doc)

        return f_list

    def _get_subtasks(self, document: Document) -> Document:
        assert document.text_representation is not None
        terms = [
            term for term in self._subtask_data["subtask_instructions"].keys() if term in document.text_representation
        ]

        for term in terms:
            subtask = self._subtask_data["subtask_instructions"][term].format(**document.properties["subtask_filters"])
            elem = Element()
            elem.text_representation = (
                subtask + "Return only the code " + term + " alongside the amount found and no additional information."
            )

            if "filters" in document.properties:
                elem.properties = {"filters": document.properties["filters"]}

            document.elements.append(elem)
        return document

    def _add_filter(self, query_body: dict, filters: dict[str, str]):
        hybrid_query_match = query_body["query"]["hybrid"]["queries"][0]
        hybrid_query_match = {
            "bool": {
                "must": [hybrid_query_match],
                "filter": [{"match_phrase": {k: filters[k]}} for k in filters],
            }
        }
        query_body["query"]["hybrid"]["queries"][0] = hybrid_query_match

        hybrid_query_knn = query_body["query"]["hybrid"]["queries"][1]
        hybrid_query_knn["knn"]["embedding"]["filter"] = {
            "bool": {"must": [{"match_phrase": {k: filters[k]}} for k in filters]}
        }
        query_body["query"]["hybrid"]["queries"][1] = hybrid_query_knn
        return query_body

    def _get_results(self, element: Element) -> Element:
        subtask = element.text_representation
        assert subtask is not None

        query = OpenSearchQuery()
        query["index"] = self._index

        qn_embedding = self._embedder.generate_text_embedding(subtask)

        query["query"] = {
            "_source": {"excludes": ["embedding"]},
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text_representation": subtask}},
                        {
                            "knn": {
                                "embedding": {
                                    "vector": qn_embedding,
                                    "k": self._os_config.get("neural_search_k", 100),
                                }
                            }
                        },
                    ]
                }
            },
            "size": self._os_config.get("size", 20),
        }

        if "llm" in self._os_config:
            query.params = {"search_pipeline": self._os_config["search_pipeline"]}
            query["query"]["ext"] = {
                "generative_qa_parameters": {
                    "llm_question": subtask,
                    "context_size": self._os_config.get("context_window", 10),
                    "llm_model": self._os_config.get("llm", "gpt-3.5-turbo"),
                }
            }
            if self._os_config.get("rerank", False):
                query["query"]["ext"]["rerank"] = {"query_context": {"query_text": subtask}}
        if "filters" in element.properties:
            query["query"] = self._add_filter(query["query"], element.properties["filters"])

        result = self._query_executor.query(query)
        result_elem = Element()
        result_elem.type = "QueryResult"
        result_elem.properties = {
            "query": result["query"],
            "hits": result["hits"],
            "generated_answer": result.generated_answer,
            "result": result["result"],
        }
        return result_elem

    def execute(self, ds: DocSet) -> list[Document]:
        # Convert docset of all questions to a docset of formulas
        formula_ds = ds.flat_map(self._get_formulas)
        subtasks_ds = formula_ds.map(self._get_subtasks)
        results_ds = subtasks_ds.map_elements(self._get_results)

        return results_ds.take_all()
