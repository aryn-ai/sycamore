from typing import Dict, Callable, List, Union, Optional

from ray.data import Dataset

from data import Element
from execution.transforms.entity import EntityExtractor, OpenAIEntityExtractor
from execution.transforms.llms import LLM
from sycamore.execution import Node, Transform


class LLMExtractEntity(Transform):
    def __init__(
        self,
        child: Node,
        entity_to_extract: Union[str, Dict],
        num_of_elements: int,
        llm: LLM,
        prompt_template: str,
        prompt_formatter: Callable[[List[Element]], str],
        entity_extractor: Optional[EntityExtractor],
        **resource_args,
    ):
        super().__init__(child, **resource_args)

        if not entity_extractor:
            entity_extractor = OpenAIEntityExtractor(
                entity_to_extract,
                num_of_elements,
                llm,
                prompt_template,
                prompt_formatter,
            )
        self.type = type
        self.entity_extractor = entity_extractor

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self.entity_extractor.extract_entity)
        return dataset
