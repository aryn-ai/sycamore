import json
from abc import ABC, abstractmethod
from typing import Union, Any, Callable, cast

from sycamore.data import Element, Document
from sycamore.execution.transforms.llms import LLM


class EntityExtractor(ABC):
    def __init__(self, entity_to_extract: Union[str, dict], llm_model: LLM):
        self._entity_to_extract = entity_to_extract
        self._model = llm_model

    @abstractmethod
    def extract_entity(self, record: dict[str, Any]) -> dict[str, Any]:
        pass


class OpenAIEntityExtractor(EntityExtractor):
    def __init__(
        self,
        entity_to_extract: Union[str, dict],
        llm_model: LLM,
        num_of_elements: int,
        prompt_template: str,
        prompt_formatter: Callable[[list[Element]], str],
    ):
        super().__init__(
            entity_to_extract,
            llm_model,
        )
        self._num_of_elements = num_of_elements
        self._prompt_template = prompt_template
        self._prompt_formatter = prompt_formatter

    def extract_entity(self, record: dict[str, Any]) -> dict[str, Any]:
        document = Document(record)

        if isinstance(self._entity_to_extract, str):
            entities = self._handle_few_shot_prompting(document)
            document.properties.update({f"{self._entity_to_extract}": entities["answer"]})
        else:
            entities = self._handle_zero_shot_prompting(document)
            for key, value in entities.items():
                document.properties.update({key: value})

        return document.to_dict()

    def _handle_few_shot_prompting(self, document: Document) -> Any:
        sub_elements = [document.elements[i] for i in range(self._num_of_elements)]

        if self._model.is_chat_mode():
            prompt = """
                {{#system~}}
                You are a helpful entity extractor.
                {{~/system}}

                {{#user~}}
                You are given a few text elements. The {{entity}} of the file is in these few text elements.Using
                this context, FIND, COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
                {{examples}}
                {{query}}
                {{~/user}}

                {{#assistant~}}
                {{gen "answer"}}
                {{~/assistant}}
                """

        else:
            prompt = """You are given a few text elements. The {{entity}} of the file is in these few text elements.Using
                    this context, FIND, COPY, and RETURN the {{entity}}. DO NOT REPHRASE OR MAKE UP AN ANSWER.
                    {{examples}}
                    {{query}}
                    =========
                    {{entity}}: {{gen "answer"}}
                    """  # noqa: E501

        entities = self._model.generate(
            prompt_kwargs={
                "prompt": prompt,
                "entity": self._entity_to_extract,
                "examples": self._prompt_template,
                "query": self._prompt_formatter(sub_elements),
            }
        )
        return entities

    def _handle_zero_shot_prompting(self, document: Document) -> Any:
        assert self._model.is_chat_mode(), (
            "Zero shot prompting is only supported for OpenAI models which " "support chat completion."
        )
        text_passage = ""
        for i in range(self._num_of_elements):
            text_passage += f"{document.elements[i]['content']['text']} "

        prompt = (
            f"Extract and save the relevant entities mentioned in the following passage together with their "
            f"properties. Only extract the properties mentioned in the 'entity_extraction' function. If a "
            f"property is not present and is not required in the function parameters, do not include it in the "
            f"output.\n"
            f"Passage:\n"
            f"{text_passage}"
        )
        llm_function = self._get_entity_extraction_function()
        llm_kwargs = self._get_llm_kwargs(llm_function)
        entities = self._model.generate(prompt_kwargs={"prompt": prompt}, llm_kwargs=llm_kwargs)
        return json.loads(entities.function_call.arguments).get("query")[0]

    def _get_entity_extraction_function(self) -> dict:
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
                    "query": {
                        "type": "array",
                        # TODO: Clean this up. We know that _entity_to_extract is a dict, becasue
                        # this is only called in the few shot case, but still don't like having to
                        # perform this cast.
                        "items": _convert_schema(cast(dict[Any, Any], self._entity_to_extract)),
                    }
                },
                "required": ["query"],
            },
        }

    def _get_llm_kwargs(self, function: dict) -> dict:
        return {"functions": [function], "function_call": {"name": function["name"]}}
