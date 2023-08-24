from enum import Enum
from typing import Dict, Optional, Any

from ray.data import Dataset

from data import Document
from sycamore.execution import (Node, Transform)
import json


class OpenAIModel(Enum):
    TEXT_DAVINCI = "text-davinci-003"
    GPT_3_5_TURBO = "gpt-3.5-turbo-0613"


class EntityExtractor:
    pass


def _get_entity_extraction_function(entity_schema: dict) -> dict:
    def _convert_schema(schema: dict) -> dict:
        props = {k: {"title": k, **v} for k, v in schema["entities"].items()}
        return {
            "type": "object",
            "properties": props,
            "required": schema.get("required", []),
        }

    return {
        "name": "entity_extraction",
        "description": "Extracts the relevant entities from the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "array", "items": _convert_schema(entity_schema)}
            },
            "required": ["query"],
        },
    }


def _get_llm_kwargs(function: dict) -> dict:
    return {"functions": [function], "function_call": {"name": function["name"]}}


class OpenAIEntityExtractor(EntityExtractor):

    def __init__(self,
                 entities_to_extract: Dict,
                 model_name: OpenAIModel,
                 num_of_elements: int = 10,
                 model_args: Optional[Dict] = None):
        self._entities_to_extract = entities_to_extract
        self._num_of_elements = num_of_elements
        self._model_name = model_name
        self._model_args = model_args

    def extract_entities(self, record: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(record)
        text_passage = ""
        for i in range(self._num_of_elements):
            text_passage += f"{document.elements[i].get('content').get('text')} "
        llm_function = _get_entity_extraction_function(self._entities_to_extract)
        llm_kwargs = _get_llm_kwargs(llm_function)
        llm_output = self._extract(text_passage, llm_kwargs)
        for key, value in llm_output.get('query')[0].items():
            document.properties.update({key: value})
        return document.to_dict()

    def _extract(self, query: str, llm_kwargs: Dict) -> Any:
        import openai
        prompt = (f"Extract and save the relevant entities mentioned in the following passage together with their "
                  f"properties. Only extract the properties mentioned in the 'information_extraction' function. If a "
                  f"property is not present and is not required in the function parameters, do not include it in the "
                  f"output."
                  f"Passage:"
                  f"{query}")

        messages = [{"role": "user", "content": f"{prompt}"}]
        completion = openai.ChatCompletion.create(model=self._model_name, messages=messages, **llm_kwargs)
        return json.loads(completion.choices[0].message.function_call.arguments)


class LLMEntityExtraction(Transform):
    def __init__(
            self,
            child: Node,
            entities_to_extract: Dict,
            num_of_elements: int,
            model_name: str,
            model_args: Optional[Dict] = None,
            **resource_args):
        super().__init__(child, **resource_args)
        self.type = type
        self.entity_extractor = OpenAIEntityExtractor(entities_to_extract,
                                                      model_name,
                                                      num_of_elements,
                                                      model_args)

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self.entity_extractor.extract_entities)
        return dataset
