import json
from typing import Optional
from sycamore.data.document import Document, OpenSearchQuery
from sycamore.data.element import Element
from sycamore.docset import DocSet
from sycamore.evaluation.data import EvaluationDataPoint
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.llms.prompts.default_prompts import TaskIdentifierZeroShotGuidancePrompt
from sycamore.transforms.embed import Embedder, SentenceTransformerEmbedder
from sycamore.transforms.query import QueryExecutor

openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
prompt = TaskIdentifierZeroShotGuidancePrompt()

class SubtaskExecutor():
    def __init__(self,
                 filepath: str,
                 index: str,
                 os_config: str,
                 query_executor: QueryExecutor,
                 embedder: Optional[Embedder] = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=100)
                 ):
        with open(filepath) as json_file:
            self._subtask_data = json.load(json_file)

        self._index = index
        self._os_config = os_config
        self._query_executor = query_executor
        self._embedder = embedder

    def _get_formulas(self, document: EvaluationDataPoint) -> list[Document]:
        f_list = []
        if document.properties["subtasks_reqd"]:
            task_id = openai_llm.generate(
                prompt_kwargs={
                    "prompt": prompt,
                    "question": document["question"],
                    "task_descriptions": self._subtask_data["task_descriptions"],
                }
            )

            formulas = self._subtask_data["task_formulas"][task_id].get("formulas", [])
            instructions = self._subtask_data["task_formulas"][task_id].get("instructions", "")

            for formula in formulas:
                doc = Document()
                doc.text_representation = formula
                doc.parent_id = document["raw"]["financebench_id"]
                doc.properties["instructions"] = instructions
                doc.properties["filters"] = document["filters"]
                f_list.append(doc)
            
            if not formulas:
                doc = Document()
                doc.text_representation = ""
                doc.parent_id = document["raw"]["financebench_id"]
                doc.properties["instructions"] = instructions
                f_list.append(doc)

        return f_list

    def _add_filter(self, query_body: dict, filters: dict[str, str]):
        hybrid_query_match = query_body["query"]["knn"]["embedding"]["filter"]["bool"]["must"]
        for key, val in filters.items():
            hybrid_query_match.append(
                {
                    "match_phrase": {
                        "properties." + key: val
                    }
                }
            )
        query_body["query"]["knn"]["embedding"]["filter"]["bool"]["must"] = hybrid_query_match
        return query_body

    def _get_subtasks(self, document: Document) -> Document:
        terms = [term for term in self._subtask_data["subtask_instructions"].keys() if term in document.text_representation]

        for term in terms:
            subtask = self._subtask_data["subtask_instructions"][term].format(**document.properties["filters"])
            sub_doc = OpenSearchQuery()
            sub_doc.index = self._index
            
            subtask += "Return only the code " + term + " alongside the amount found and no additional information."
            qn_embedding = self._embedder.generate_text_embeddings(subtask)

            sub_doc.query = {
                "_source": {"excludes": ["embedding"]},
                "size": self._os_config.get("size", 20),
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": qn_embedding,
                            "k": self._os_config.get("neural_search_k", 100),
                            "filter": {
                                "bool": {
                                    "must": [
                                        {
                                            "match": {
                                                "text_representation": subtask
                                            }
                                        },
                                    ],
                                },
                            }
                        }
                    }
                }
            }

            if "llm" in self._os_config:
                sub_doc.params = {"search_pipeline": self._os_config["search_pipeline"]}
                sub_doc["query"]["ext"] = {
                    "generative_qa_parameters": {
                        "llm_question": subtask,
                        "context_size":self. _os_config.get("context_window", 10),
                        "llm_model": self._os_config.get("llm", "gpt-3.5-turbo"),
                    }
                }
                if self._os_config.get("rerank", False):
                    sub_doc["query"]["ext"]["rerank"] = {"query_context": {"query_text": subtask}}
            if "filters" in document.properties:
                sub_doc["query"] = self._add_filter(sub_doc["query"], document.properties["filters"])

            document.elements.append(sub_doc)
        return document

    def _get_results(self, element: Element) -> Element:
        return self._query_executor.query(element)

    def execute(self, ds: DocSet) -> DocSet:
        # Convert docset of all questions to a docset of formulas
        formula_ds = ds.flat_map(self._get_formulas)
        subtasks_ds = formula_ds.map(self._get_subtasks)
        results_ds = subtasks_ds.map_elements(self._get_results)

        return results_ds.take_all()