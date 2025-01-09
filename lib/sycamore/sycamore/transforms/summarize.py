import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional


from sycamore.data import Element, Document
from sycamore.functions import Tokenizer, CharacterTokenizer
from sycamore.llms.prompts.default_prompts import SummarizeDataMessagesPrompt
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.llms.prompts import TextSummarizerGuidancePrompt
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace

NUM_DOCS_GENERATE = 60
NUM_TEXT_CHARS_GENERATE = 2500
BASE_PROPS = [
    "filename",
    "filetype",
    "page_number",
    "page_numbers",
    "links",
    "element_id",
    "parent_id",
    "_schema",
    "_schema_class",
    "entity",
]


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, document: Document) -> Document:
        pass


class LLMElementTextSummarizer(Summarizer):
    """
    LLMElementTextSummarizer uses a specified LLM to summarize text data within elements of a document.

    Args:
        llm: An instance of an LLM class to use for text summarization.
        element_operator: A callable function that operates on the document and returns a list of elements to be
            summarized. Default is None.

    Example:
         .. code-block:: python

            llm_model = OpenAILanguageModel("gpt-3.5-turbo")
            element_operator = my_element_selector  # A custom element selection function
            summarizer = LLMElementTextSummarizer(llm_model, element_operator)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .summarize(summarizer=summarizer)
    """

    def __init__(self, llm: LLM, element_operator: Optional[Callable[[Element], bool]] = None):
        self._llm = llm
        self._element_operator = element_operator

    def summarize(self, document: Document) -> Document:
        elements = []
        if self._element_operator is not None:
            for element in document.elements:
                if self._element_operator(element):
                    elements.append(self._summarize_text_element(element))
                else:
                    elements.append(element)
        else:
            elements = [self._summarize_text_element(element) for element in document.elements]

        document.elements = elements
        return document

    @timetrace("SummText")
    def _summarize_text_element(self, element: Element) -> Element:
        prompt = TextSummarizerGuidancePrompt()

        if element.text_representation:
            response = self._llm.generate(prompt_kwargs={"prompt": prompt, "query": element.text_representation})
            element.properties["summary"] = response
        return element


class QuestionAnsweringSummarizer:
    def __init__(self, llm: LLM, question: str):
        self.llm = llm
        self.question = question

    def __call__(self, text: str) -> str:
        messages = SummarizeDataMessagesPrompt(question=self.question, text=text).as_messages()
        prompt_kwargs = {"messages": messages}

        t0 = time.time()
        # call to LLM
        summary = self.llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})
        t1 = time.time()
        logging.info(f"Summarizer took {t1 - t0} seconds to generate summary.")

        return summary


def collapse(text: str, tokens_per_chunk: int, tokenizer: Tokenizer, summarizer_fn: Callable[[str], str]) -> str:
    """
    Collapses text iteratively, summarizing the first chunk and incorporating it in the summary for the next chunk.

    Args:
        text: Text to collapse.
        chunk_size: Size of each chunk.
        tokenizer: Tokenizer to use for counting against max_tokens.

    Returns:
        List of chunks.
    """
    tokens = tokenizer.tokenize(text)
    total = len(tokens)
    if total <= tokens_per_chunk:
        return text
    done = False
    i = 0
    additional = i + tokens_per_chunk
    cur_summary = ""
    while not done:
        input = ""
        if cur_summary:
            input = f"{cur_summary}\n"
        input += "".join([str(tk) for tk in tokens[i : i + additional]])  # make mypy happy
        print(f"input size: {len(input)}")
        cur_summary = summarizer_fn(input)
        assert (
            len(cur_summary) <= tokens_per_chunk
        ), f"Summarizer output is longer than input chunk {len(cur_summary)} > {tokens_per_chunk} !!!"
        print(f"summary to chunk ratio: {len(cur_summary) / tokens_per_chunk}")
        i += additional
        remaining = tokens_per_chunk - len(cur_summary)
        additional = min(remaining, total - i)
        if additional == 0:
            break

    return cur_summary


class DocumentSummarizer(Summarizer):
    def __init__(
        self,
        llm: LLM,
        question: str,
        chunk_size: int = 10 * 1000,
        tokenizer: Tokenizer = CharacterTokenizer(),
        chunk_overlap: int = 0,
        use_elements: bool = False,
        num_elements: int = 5,
    ):
        self.llm = llm
        self.question = question
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.chunk_overlap = chunk_overlap
        self.use_elements = use_elements
        self.num_elements = num_elements

    def summarize(self, document: Document) -> Document:
        text = self.get_text(document)
        summary = collapse(text, self.chunk_size, self.tokenizer, QuestionAnsweringSummarizer(self.llm, self.question))
        document.properties["summary"] = summary
        return document

    def get_text(self, doc: Document) -> str:
        doc_text = ""
        props_dict = doc.properties.get("entity", {})
        props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
        for k, v in props_dict.items():
            doc_text += f"{k}: {v}\n"

        doc_text_representation = ""
        if not self.use_elements:
            if doc.text_representation is not None:
                doc_text_representation += doc.text_representation[:NUM_TEXT_CHARS_GENERATE]
        else:
            for element in doc.elements[: self.num_elements]:
                # Greedy fill doc level text length
                if len(doc_text_representation) >= NUM_TEXT_CHARS_GENERATE:
                    break
                doc_text_representation += (element.text_representation or "") + "\n"
        doc_text += f"Text contents:\n{doc_text_representation}\n"

        return doc_text


class Summarize(NonCPUUser, NonGPUUser, Map):
    """
    The summarize transform generates summaries of documents or elements.
    """

    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, f=summarizer.summarize, **kwargs)
