from collections.abc import Mapping
import logging
import pprint
import sys
from typing import Callable, Optional, Any, Iterable, Type

from sycamore import Context
from sycamore.data import Document, Element, MetadataDocument
from sycamore.functions.tokenizer import Tokenizer
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.augment_text import TextAugmentor
from sycamore.transforms.embed import Embedder
from sycamore.transforms.extract_entity import EntityExtractor
from sycamore.transforms.extract_schema import SchemaExtractor, PropertyExtractor
from sycamore.transforms.partition import Partitioner
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.extract_table import TableExtractor
from sycamore.transforms.merge_elements import ElementMerger
from sycamore.writer import DocSetWriter
from sycamore.transforms.query import QueryExecutor, Query

logger = logging.getLogger(__name__)


class DocSet:
    """
    A DocSet, short for “documentation set,” is a distributed collection of documents bundled together for processing.
    Sycamore provides a variety of transformations on DocSets to help customers handle unstructured data easily.
    """

    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def lineage(self) -> Node:
        return self.plan

    def explain(self) -> None:
        # TODO, print out nice format DAG
        pass

    def show(
        self,
        limit: int = 20,
        show_elements: bool = True,
        num_elements: int = -1,  # -1 shows all elements
        show_binary: bool = False,
        show_embedding: bool = False,
        truncate_content: bool = True,
        truncate_length: int = 100,
        stream=sys.stdout,
    ) -> None:
        """
        Prints the content of the docset in a human-readable format. It is useful for debugging and
        inspecting the contents of objects during development.

        Args:
            limit: The maximum number of items to display.
            show_elements: Whether to display individual elements or not.
            num_elements: The number of elements to display. Use -1 to show all elements.
            show_binary: Whether to display binary data or not.
            show_embedding: Whether to display embedding information or not.
            truncate_content: Whether to truncate long content when displaying.
            truncate_length: The maximum length of content to display when truncating.
            stream: The output stream where the information will be displayed.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .show()
        """
        documents = self.take(limit)

        def _truncate(s):
            if len(s) <= truncate_length:
                return s
            amount_truncated = len(s) - truncate_length
            return s[:truncate_length] + f" <{amount_truncated} chars>"

        for document in documents:
            if not show_elements:
                num_elems = len(document.elements)
                document.data["elements"] = f"<{num_elems} elements>"
            else:
                if document.elements is not None and 0 <= num_elements < len(document.elements):
                    document.elements = document.elements[:num_elements]

            if not show_binary and document.binary_representation is not None:
                binary_length = len(document.binary_representation)
                document.binary_representation = f"<{binary_length} bytes>".encode("utf-8")

            if truncate_content and document.text_representation is not None:
                document.text_representation = _truncate(document.text_representation)

            if not show_embedding and document.embedding is not None:
                embedding_length = len(document.embedding)
                document.data["embedding"] = f"<{embedding_length} floats>"

            if show_elements and "elements" in document.data:
                if not show_binary:
                    for i, e in enumerate(document.data["elements"]):
                        if e.get("binary_representation") is not None:
                            binary_length = len(e["binary_representation"])
                            e["binary_representation"] = f"<{binary_length} bytes>".encode("utf-8")
                if truncate_content:
                    for i, e in enumerate(document.data["elements"]):
                        if e.get("text_representation") is not None:
                            e["text_representation"] = _truncate(e["text_representation"])

            pprint.pp(document, stream=stream)

    def count(self, include_metadata=False, **execution_args) -> int:
        """
        Counts the number of documents in the resulting dataset.
        It is a convenient way to determine the size of the dataset generated by the plan.

        Returns:
            The number of documents in the docset.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .count()
        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **execution_args)
        # We could parallelize like ray dataset.count() or optimize the metadata case, but it's not worth the complexity
        count = 0
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if not include_metadata and isinstance(doc, MetadataDocument):
                continue
            count = count + 1

        return count

    def take(self, limit: int = 20, include_metadata: bool = False, **execution_args) -> list[Document]:
        """
        Returns up to ``limit`` documents from the dataset.

        Args:
            limit: The maximum number of Documents to return.

        Returns:
            A list of up to ``limit`` Documents from the Docset.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .take()

        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **execution_args)
        ret = []
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if not include_metadata and isinstance(doc, MetadataDocument):
                continue
            ret.append(doc)
            if len(ret) >= limit:
                break

        return ret

    def take_all(self, limit: Optional[int] = None, include_metadata: bool = False, **execution_args) -> list[Document]:
        """
        Returns all of the rows in this DocSet.

        If limit is set, this method will raise an error if this Docset
        has more than `limit` Documents, including metadata.

        Args:
            limit: The number of Documents above which this method will raise an error.
        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **execution_args)
        docs = [Document.from_row(row) for row in dataset.take_all(limit)]
        if include_metadata:
            return docs
        else:
            return [d for d in docs if not isinstance(d, MetadataDocument)]

    def limit(self, limit: int = 20, **kwargs) -> "DocSet":
        """
        Applies the Limit transforms on the Docset.

        Args:
            limit: The maximum number of documents to include in the resulting Docset.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .explode()
                    .limit()

        """
        from sycamore.transforms import Limit

        return DocSet(self.context, Limit(self.plan, limit, **kwargs))

    def partition(
        self, partitioner: Partitioner, table_extractor: Optional[TableExtractor] = None, **kwargs
    ) -> "DocSet":
        """
        Applies the Partition transform on the Docset.

        More information can be found in the :ref:`Partition` documentation.

        Example:
            .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
        """
        from sycamore.transforms import Partition

        plan = Partition(self.plan, partitioner=partitioner, table_extractor=table_extractor, **kwargs)
        return DocSet(self.context, plan)

    def with_property(self, name, f: Callable[[Document], Any], **resource_args) -> "DocSet":
        """
        Applies a function to each document and adds the result as a property.

        Args:
            name: The name of the property to add to each Document.
            f: The function to apply to each Document.

        Example:
             To add a property that contains the length of the text representation as a new property:
             .. code-block:: python

                docset.with_property("text_size", lambda doc: len(doc.text_representation))
        """
        return self.with_properties({name: f}, **resource_args)

    def with_properties(self, property_map: Mapping[str, Callable[[Document], Any]], **resource_args) -> "DocSet":
        """
        Adds multiple properties to each Document.

        Args:
            property_map: A mapping of property names to functions to generate those properties

        Example:
            .. code-block:: python

               docset.with_properties({
                   "text_size": lambda doc: len(doc.text_representation),
                   "truncated_text": lambda doc: doc.text_representation[0:256]
               })
        """

        def add_properties_fn(doc: Document) -> Document:
            doc.properties.update({k: f(doc) for k, f in property_map.items()})
            return doc

        return self.map(add_properties_fn, **resource_args)

    def spread_properties(self, props: list[str], **resource_args) -> "DocSet":
        """
        Copies listed properties from parent document to child elements.

        Example:
            .. code-block:: python

               pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .spread_properties(["title"])
                    .explode()
        """
        from sycamore.transforms import SpreadProperties

        plan = SpreadProperties(self.plan, props, **resource_args)
        return DocSet(self.context, plan)

    def augment_text(self, augmentor: TextAugmentor, **resource_args) -> "DocSet":
        """
        Augments text_representation with external information.

        Args:
            augmentor (TextAugmentor): A TextAugmentor instance that defines how to augment the text

        Example:
         .. code-block:: python

            augmentor = FStringTextAugmentor(sentences = [
                "This pertains to the part {doc.properties['part_name']}.",
                "{doc.text_representation}"
            ])
            entity_extractor = OpenAIEntityExtractor("part_name",
                                        llm=openai_llm,
                                        prompt_template=part_name_template)
            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=SycamorePartitioner())
                .extract_entity(entity_extractor)
                .explode()
                .augment_text(augmentor)
        """
        from sycamore.transforms.augment_text import AugmentText

        plan = AugmentText(self.plan, augmentor, **resource_args)
        return DocSet(self.context, plan)

    def split_elements(self, tokenizer: Tokenizer, max_tokens: int = 512, **kwargs) -> "DocSet":
        """
        Splits elements if they are larger than the maximum number of tokens.

        Example:
            .. code-block:: python

               pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .split_elements(tokenizer=tokenizer, max_tokens=512)
                    .explode()
        """
        from sycamore.transforms import SplitElements

        plan = SplitElements(self.plan, tokenizer, max_tokens, **kwargs)
        return DocSet(self.context, plan)

    def explode(self, **resource_args) -> "DocSet":
        """
        Applies the Explode transform on the Docset.

        Example:
            .. code-block:: python

                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .explode()
        """
        from sycamore.transforms.explode import Explode

        explode = Explode(self.plan, **resource_args)
        return DocSet(self.context, explode)

    def embed(self, embedder: Embedder, **kwargs) -> "DocSet":
        """
        Applies the Embed transform on the Docset.

        Args:
            embedder: An instance of an Embedder class that defines the embedding method to be applied.


        Example:
            .. code-block:: python

                model_name="sentence-transformers/all-MiniLM-L6-v2"
                embedder = SentenceTransformerEmbedder(batch_size=100, model_name=model_name)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .explode()
                    .embed(embedder=embedder)
        """
        from sycamore.transforms import Embed

        embeddings = Embed(self.plan, embedder=embedder, **kwargs)
        return DocSet(self.context, embeddings)

    def extract_entity(self, entity_extractor: EntityExtractor, **kwargs) -> "DocSet":
        """
        Applies the ExtractEntity transform on the Docset.

        Args:
            entity_extractor: An instance of an EntityExtractor class that defines the entity extraction method to be
                applied.

        Example:
             .. code-block:: python

                 title_context_template = "template"

                 openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
                 entity_extractor = OpenAIEntityExtractor("title",
                                        llm=openai_llm,
                                        prompt_template=title_context_template)

                 context = sycamore.init()
                 pdf_docset = context.read.binary(paths, binary_format="pdf")
                     .partition(partitioner=SycamorePartitioner())
                     .extract_entity(entity_extractor=entity_extractor)

        """
        from sycamore.transforms import ExtractEntity

        entities = ExtractEntity(self.plan, entity_extractor=entity_extractor, **kwargs)
        return DocSet(self.context, entities)

    def extract_schema(self, schema_extractor: SchemaExtractor, **kwargs) -> "DocSet":
        """
        Extracts a JSON schema of extractable properties from each document in this DocSet.

        Each schema is a mapping of names to types that corresponds to fields that are present in the document.
        For example, calling this method on a financial document containing information about companies
        might yield a schema like

        .. code-block:: python

            {
              "company_name": "string",
              "revenue": "number",
              "CEO": "string"
            }

        This method will extract a unique schema for each document in the DocSet independently.
        If the documents in the DocSet represent instances with a common schema, consider
        `ExtractBatchSchema` which will extract a common schema for all documents.

        The dataset is returned with an additional `_schema` property that contains JSON-encoded schema, if any
        is detected.

        Args:
            schema_extractor: A `SchemaExtractor` instance to extract the schema for each document.

        Example:
            .. code-block:: python

                openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
                schema_extractor=OpenAISchemaExtractor("Corporation", llm=openai, num_of_elements=35)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .extract_schema(schema_extractor=schema_extractor)
        """

        from sycamore.transforms import ExtractSchema

        schema = ExtractSchema(self.plan, schema_extractor=schema_extractor)
        return DocSet(self.context, schema)

    def extract_batch_schema(self, schema_extractor: SchemaExtractor, **kwargs) -> "DocSet":
        """
        Extracts a common schema from the documents in this DocSet.

        This transform is similar to extract_schema, except that it will add the same schema
        to each document in the DocSet rather than infering a separate schema per Document.
        This is most suitable for document collections that share a common format. If you have
        a heterogeneous document collection and want a different schema for each type, consider
        using extract_schema instead.

        Args:
            schema_extractor: A `SchemaExtractor` instance to extract the schema for each document.

        Example:
            .. code-block:: python

                openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
                schema_extractor=OpenAISchemaExtractor("Corporation", llm=openai, num_of_elements=35)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .extract_batch_schema(schema_extractor=schema_extractor)
        """

        from sycamore.transforms import ExtractBatchSchema

        schema = ExtractBatchSchema(self.plan, schema_extractor=schema_extractor)
        return DocSet(self.context, schema)

    def extract_properties(self, property_extractor: PropertyExtractor, **kwargs) -> "DocSet":
        """
        Extracts properties from each Document in this DocSet based on the `_schema` property.

        The schema can be computed using `extract_schema` or `extract_batch_schema` or can be
        provided manually in JSON-schema format in the `_schema` field under `Document.properties`.


        Example:
            .. code-block:: python

                openai_llm = OpenAI(OpenAIModels.GPT_3_5_TURBO.value)
                property_extractor = OpenAIPropertyExtractor(OpenaAIPropertyExtrator(llm=openai_llm))

                context = sycamore.init()

                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partition=SycamorePartitioner())
                    .extract_properties(property_extractor)
        """
        from sycamore.transforms import ExtractProperties

        schema = ExtractProperties(self.plan, property_extractor=property_extractor)
        return DocSet(self.context, schema)

    def summarize(self, summarizer: Summarizer, **kwargs) -> "DocSet":
        """
        Applies the Summarize transform on the Docset.

        Example:
            .. code-block:: python

                llm_model = OpenAILanguageModel("gpt-3.5-turbo")
                element_operator = my_element_selector  # A custom element selection function
                summarizer = LLMElementTextSummarizer(llm_model, element_operator)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .summarize(summarizer=summarizer)
        """
        from sycamore.transforms import Summarize

        summaries = Summarize(self.plan, summarizer=summarizer, **kwargs)
        return DocSet(self.context, summaries)

    def mark_bbox_preset(self, tokenizer: Tokenizer, token_limit: int = 512, **kwargs) -> "DocSet":
        """
        Convenience composition of:
            SortByPageBbox
            MarkDropTiny minimum=2
            MarkDropHeaderFooter top=0.05 bottom=0.05
            MarkBreakPage
            MarkBreakByColumn
            MarkBreakByTokens limit=512
        Meant to work in concert with MarkedMerger.
        """
        from sycamore.transforms import (
            SortByPageBbox,
            MarkDropTiny,
            MarkDropHeaderFooter,
            MarkBreakPage,
            MarkBreakByColumn,
            MarkBreakByTokens,
        )

        plan0 = SortByPageBbox(self.plan, **kwargs)
        plan1 = MarkDropTiny(plan0, 2, **kwargs)
        plan2 = MarkDropHeaderFooter(plan1, 0.05, 0.05, **kwargs)
        plan3 = MarkBreakPage(plan2, **kwargs)
        plan4 = MarkBreakByColumn(plan3, **kwargs)
        plan5 = MarkBreakByTokens(plan4, tokenizer, token_limit, **kwargs)
        return DocSet(self.context, plan5)

    def merge(self, merger: ElementMerger, **kwargs) -> "DocSet":
        """
        Applies merge operation on each list of elements of the Docset

        Example:
             .. code-block:: python

                from transformers import AutoTokenizer
                tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                merger = GreedyElementMerger(tk, 512)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .merge(merger=merger)
        """
        from sycamore.transforms import Merge

        merged = Merge(self.plan, merger=merger, **kwargs)
        return DocSet(self.context, merged)

    def regex_replace(self, spec: list[tuple[str, str]], **kwargs) -> "DocSet":
        """
        Performs regular expression replacement (using re.sub()) on the
        text_representation of every Element in each Document.

        Example:
            .. code-block:: python

               from sycamore.transforms import COALESCE_WHITESPACE
               ds = context.read.binary(paths, binary_format="pdf")
                   .partition(partitioner=SycamorePartitioner())
                   .regex_replace(COALESCE_WHITESPACE)
                   .regex_replace([(r"\d+", "1313"), (r"old", "new")])
                   .explode()
        """
        from sycamore.transforms import RegexReplace

        plan = RegexReplace(self.plan, spec, **kwargs)
        return DocSet(self.context, plan)

    def sketch(self, window: int = 17, number: int = 16, **kwargs) -> "DocSet":
        """
        For each Document, uses shingling to hash sliding windows of the
        text_representation.  The set of shingles is called the sketch.
        Documents' sketches can be compared to determine if they have
        near-duplicate content.

        Args:
            window: Number of bytes in the sliding window that is hashed (17)
            number: Count of hashes comprising a shingle (16)

        Example:
            .. code-block:: python

               ds = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .explode()
                    .sketch(window=17)
        """
        from sycamore.transforms import Sketcher

        plan = Sketcher(self.plan, window=window, number=number, **kwargs)
        return DocSet(self.context, plan)

    def term_frequency(self, tokenizer: Tokenizer, with_token_ids: bool = False, **kwargs) -> "DocSet":
        """
        For each document, compute a frequency table over the text representation, as
        tokenized by `tokenizer`. Use to enable hybrid search in Pinecone

        Example:
            .. code-block:: python

                tk = OpenAITokenizer("gpt-3.5-turbo")
                context = sycamore.init()
                context.read.binary(paths, binary_format="pdf")
                    .partition(SycamorePartitioner())
                    .explode()
                    .term_frequency(tokenizer=tk)
                    .show()
        """
        from sycamore.transforms import TermFrequency

        plan = TermFrequency(self.plan, tokenizer=tokenizer, with_token_ids=with_token_ids, **kwargs)
        return DocSet(self.context, plan)

    def transform(self, cls: Type[Transform], **kwargs) -> "DocSet":
        """
        Add specified transform class to pipeline.  See the API
        reference section on transforms.

        Args:
            cls: Class of transform to instantiate into pipeline
            ...: Other keyword arguments are passed to class constructor

        Example:
            .. code-block:: python


               from sycamore.transforms import FooBar
               ds = context.read.binary(paths, binary_format="pdf")
                   .partition(partitioner=SycamorePartitioner())
                   .transform(cls=FooBar, arg=123)
        """
        plan = cls(self.plan, **kwargs)  # type: ignore
        return DocSet(self.context, plan)

    def map(self, f: Callable[[Document], Document], **resource_args) -> "DocSet":
        """
        Applies the Map transformation on the Docset.

        Args:
            f: The function to apply to each document.

        """
        from sycamore.transforms import Map

        mapping = Map(self.plan, f=f, **resource_args)
        return DocSet(self.context, mapping)

    def flat_map(self, f: Callable[[Document], list[Document]], **resource_args) -> "DocSet":
        """
        Applies the FlatMap transformation on the Docset.

        Args:
            f: The function to apply to each document.

        Example:
             .. code-block:: python

                def custom_flat_mapping_function(document: Document) -> list[Document]:
                    # Custom logic to transform the document and return a list of documents
                    return [transformed_document_1, transformed_document_2]

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=SycamorePartitioner())
                .flat_map(custom_flat_mapping_function)

        """
        from sycamore.transforms import FlatMap

        flat_map = FlatMap(self.plan, f=f, **resource_args)
        return DocSet(self.context, flat_map)

    def filter(self, f: Callable[[Document], bool], **resource_args) -> "DocSet":
        """
        Applies the Filter transform on the Docset.

        Args:
            f: A callable function that takes a Document object and returns a boolean indicating whether the document
                should be included in the filtered Docset.

        Example:
             .. code-block:: python

                def custom_filter(doc: Document) -> bool:
                    # Define your custom filtering logic here.
                    return doc.some_property == some_value

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())
                    .filter(custom_filter)

        """
        from sycamore.transforms import Filter

        filtered = Filter(self.plan, f=f, **resource_args)
        return DocSet(self.context, filtered)

    def filter_elements(self, f: Callable[[Element], bool], **resource_args) -> "DocSet":
        """
        Applies the given filter function to each element in each Document in this DocsSet.

        Args:
            f: A Callable that takes an Element and returns True if the element should be retained.
        """

        def process_doc(doc: Document) -> Document:
            new_elements = [e for e in doc.elements if f(e)]
            doc.elements = new_elements
            return doc

        return self.map(process_doc, **resource_args)

    def map_batch(
        self,
        f: Callable[[list[Document]], list[Document]],
        f_args: Optional[Iterable[Any]] = None,
        f_kwargs: Optional[dict[str, Any]] = None,
        f_constructor_args: Optional[Iterable[Any]] = None,
        f_constructor_kwargs: Optional[dict[str, Any]] = None,
        **resource_args,
    ) -> "DocSet":
        """
        The map_batch transform is similar to map, except that it processes a list of documents and returns a list of
        documents. map_batch is ideal for transformations that get performance benefits from batching.

        Example:
             .. code-block:: python

                def custom_map_batch_function(documents: list[Document]) -> list[Document]:
                    # Custom logic to transform the documents
                    return transformed_documents

                map_ds = input_ds.map_batch(f=custom_map_batch_function)

                def CustomMappingClass():
                    def __init__(self, arg1, arg2, *, kw_arg1=None, kw_arg2=None):
                        self.arg1 = arg1
                        # ...

                    def _process(self, doc: Document) -> Document:
                        doc.properties["arg1"] = self.arg1
                        return doc

                    def __call__(self, docs: list[Document], fnarg1, *, fnkwarg1=None) -> list[Document]:
                        return [self._process(d) for d in docs]

                map_ds = input_ds.map_batch(f=CustomMappingClass,
                                            f_args=["fnarg1"], f_kwargs={"fnkwarg1": "stuff"},
                                            f_constructor_args=["arg1", "arg2"],
                                            f_constructor_kwargs={"kw_arg1": 1, "kw_arg2": 2})
        """
        from sycamore.transforms import MapBatch

        map_batch = MapBatch(
            self.plan,
            f=f,
            f_args=f_args,
            f_kwargs=f_kwargs,
            f_constructor_args=f_constructor_args,
            f_constructor_kwargs=f_constructor_kwargs,
            **resource_args,
        )
        return DocSet(self.context, map_batch)

    def map_elements(self, f: Callable[[Element], Element], **resource_args) -> "DocSet":
        """
        Applies the given mapping function to each element in the each Document in this DocsSet.

        Args:
            f: A Callable that takes an Element and returns an Element. Elements for which
                f evaluates to None are dropped.
        """

        def process_doc(doc: Document) -> Document:
            new_elements = []
            for e in doc.elements:
                new_element = f(e)
                if new_element is not None:
                    new_elements.append(new_element)
            doc.elements = new_elements
            return doc

        return self.map(process_doc, **resource_args)

    def random_sample(self, fraction: float, seed: Optional[int] = None) -> "DocSet":
        """
        Retain a random sample of documents from this DocSet.

        The number of documents in the output will be approximately `fraction * self.count()`

        Args:
            fraction: The fraction of documents to retain.
            seed: Optional seed to use for the RNG.

        """
        from sycamore.transforms import RandomSample

        sampled = RandomSample(self.plan, fraction=fraction, seed=seed)
        return DocSet(self.context, sampled)

    def query(self, query_executor: QueryExecutor, **resource_args) -> "DocSet":
        """
        Applies a query execution transform on a DocSet of queries.

        Args:
            query_executor: Implementation for the query execution.

        """

        query = Query(self.plan, query_executor, **resource_args)
        return DocSet(self.context, query)

    @property
    def write(self) -> DocSetWriter:
        """
        Exposes an interface for writing a DocSet to OpenSearch or other external storage.
        See :class:`~writer.DocSetWriter` for more information about writers and their arguments.

        Example:
             The following example shows reading a DocSet from a collection of PDFs, partitioning
             it using the ``SycamorePartitioner``, and then writing it to a new OpenSearch index.

             .. code-block:: python

                os_client_args = {
                    "hosts": [{"host": "localhost", "port": 9200}],
                    "http_auth": ("user", "password"),
                }

                index_settings = {
                    "body": {
                        "settings": {
                            "index.knn": True,
                        },
                        "mappings": {
                            "properties": {
                                "embedding": {
                                    "type": "knn_vector",
                                    "dimension": 384,
                                    "method": {"name": "hnsw", "engine": "faiss"},
                                },
                            },
                        },
                    },
                }

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=SycamorePartitioner())

                pdf.write.opensearch(
                     os_client_args=os_client_args,
                     index_name="my_index",
                     index_settings=index_settings)
        """
        return DocSetWriter(self.context, self.plan)
