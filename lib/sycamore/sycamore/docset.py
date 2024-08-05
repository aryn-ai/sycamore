from collections.abc import Mapping
import logging
from pathlib import Path
import pprint
import sys
from typing import Callable, Optional, Any, Iterable, Type, Union

from sycamore.context import Context
from sycamore.data import Document, Element, MetadataDocument
from sycamore.functions.tokenizer import Tokenizer
from sycamore.llms.llms import LLM
from sycamore.llms.prompts.default_prompts import (
    LlmClusterEntityAssignGroupsMessagesPrompt,
    LlmClusterEntityFormGroupsMessagesPrompt,
)
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.augment_text import TextAugmentor
from sycamore.transforms.embed import Embedder
from sycamore.transforms.extract_entity import EntityExtractor, OpenAIEntityExtractor
from sycamore.transforms.extract_graph import GraphExtractor
from sycamore.transforms.extract_schema import SchemaExtractor, PropertyExtractor
from sycamore.transforms.partition import Partitioner
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.llm_query import LLMTextQueryAgent
from sycamore.transforms.extract_table import TableExtractor
from sycamore.transforms.merge_elements import ElementMerger
from sycamore.utils.extract_json import extract_json
from sycamore.writer import DocSetWriter
from sycamore.transforms.query import QueryExecutor, Query

logger = logging.getLogger(__name__)


class DocSet:
    """
    A DocSet, short for “Document Set”, is a distributed collection of documents bundled together for processing.
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
                    .partition(partitioner=ArynPartitioner())
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

    def count(self, include_metadata=False, **kwargs) -> int:
        """
        Counts the number of documents in the resulting dataset.
        It is a convenient way to determine the size of the dataset generated by the plan.

        Args:
            include_metadata: Determines whether or not to count MetaDataDocuments
            **kwargs

        Returns:
            The number of documents in the docset.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=ArynPartitioner())
                    .count()
        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **kwargs)
        # We could parallelize like ray dataset.count() or optimize the metadata case, but it's not worth the complexity
        count = 0
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if not include_metadata and isinstance(doc, MetadataDocument):
                continue
            count = count + 1

        return count

    def count_distinct(self, field: str, **kwargs) -> int:
        """
        Counts the number of documents in the resulting dataset with a unique
        value for field.

        Args:
        field: Field (in dotted notation) to perform a unique count based on.
        **kwargs

        Returns:
            The number of documents with a unique value for field.

        Example:
             .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=ArynPartitioner())
                    .count("doc_id")
        """
        from sycamore import Execution

        unique_docs = set()
        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **kwargs)
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if isinstance(doc, MetadataDocument):
                continue
            value = doc.field_to_value(field)
            if value is not None and value != "None":
                unique_docs.add(value)
        return len(unique_docs)

    def take(self, limit: int = 20, include_metadata: bool = False, **kwargs) -> list[Document]:
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
                    .partition(partitioner=ArynPartitioner())
                    .take()

        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        ret = []
        for doc in execution.execute_iter(self.plan, **kwargs):
            if not include_metadata and isinstance(doc, MetadataDocument):
                continue
            ret.append(doc)
            if len(ret) >= limit:
                break

        return ret

    def take_all(self, limit: Optional[int] = None, include_metadata: bool = False, **kwargs) -> list[Document]:
        """
        Returns all of the rows in this DocSet.

        If limit is set, this method will raise an error if this Docset
        has more than `limit` Documents, including metadata.

        Args:
            limit: The number of Documents above which this method will raise an error.
        """
        from sycamore import Execution

        execution = Execution(self.context, self.plan)
        dataset = execution.execute(self.plan, **kwargs)
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                     .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
                    .extract_batch_schema(schema_extractor=schema_extractor)
        """

        from sycamore.transforms import ExtractBatchSchema

        schema = ExtractBatchSchema(self.plan, schema_extractor=schema_extractor)
        return DocSet(self.context, schema)

    def extract_graph_structure(self, extractors: list[GraphExtractor], **kwargs) -> "DocSet":
        """
        Extracts metadata from documents into a format that sets up resulting docset to be loaded into neo4j

        Args:
            extractors: A list of GraphExtractor objects which determine what is extracted from the docset

        Example:
            .. code-block:: python

                metadata = [GraphMetadata(nodeKey='company',nodeLabel='Company',relLabel='FILED_BY'),
                GraphMetadata(nodeKey='gics_sector',nodeLabel='Sector',relLabel='IN_SECTOR'),
                GraphMetadata(nodeKey='doc_type',nodeLabel='Document Type',relLabel='IS_TYPE'),
                GraphMetadata(nodeKey='doc_period',nodeLabel='Year',relLabel='FILED_DURING'),
                ]

                ds = (
                    ctx.read.manifest(metadata_provider=JsonManifestMetadataProvider(manifest),...)
                    .partition(partitioner=ArynPartitioner(...), num_gpus=0.1)
                    .extract_graph_structure(extractors=[MetadataExtractor(metadata=metadata)])
                    .explode()
                )
        """
        docset = self
        for extractor in extractors:
            docset = extractor.extract(docset)

        return docset

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
                    .partition(partition=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                   .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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
                    .partition(ArynPartitioner())
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
                   .partition(partitioner=ArynPartitioner())
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
                .partition(partitioner=ArynPartitioner())
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
                    .partition(partitioner=ArynPartitioner())
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

    def llm_filter(
        self,
        llm: LLM,
        new_field: str,
        prompt: Union[list[dict], str],
        field: Optional[str] = "text_representation",
        threshold: int = 3,
        **resource_args,
    ) -> "DocSet":
        """
        Filters DocSet to only keep documents that score (determined by LLM) greater
        than or equal to the inputted threshold value.

        Args:
            client: LLM client to use.
            new_field: The field that will be added to the DocSet with the outputs.
            prompt: LLM prompt.
            field: Document field to filter based on.
            threshold: Cutoff that determines whether or not to keep document.
            **resource_args

        Returns:
            A filtered DocSet.
        """

        def threshold_filter(doc: Document, threshold) -> bool:
            try:
                return_value = int(doc.properties[new_field]) >= threshold
            except Exception:
                # accounts for llm output errors
                return_value = False

            return return_value

        docset = self.filter(lambda doc: doc.field_to_value(field) is not None and doc.field_to_value(field) != "None")

        entity_extractor = OpenAIEntityExtractor(
            entity_name=new_field, llm=llm, use_elements=False, prompt=prompt, field=field
        )
        docset = docset.extract_entity(entity_extractor=entity_extractor)
        docset = docset.filter(lambda doc: threshold_filter(doc, threshold), **resource_args)

        return docset

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

    def sort(self, descending: bool, field: str, default_val: Optional[Any] = None) -> "DocSet":
        """
        Sort DocSet by specified field.

        Args:
            descending: Whether or not to sort in descending order (first to last).
            field: Document field in relation to Document using dotted notation, e.g. properties.filetype
            default_val: Default value to use if field does not exist in Document
        """
        from sycamore.transforms import Sort

        return DocSet(self.context, Sort(self.plan, descending, field, default_val))

    def groupby_count(self, field: str, unique_field: Optional[str] = None, **kwargs) -> "DocSet":
        """
        Performs a count aggregation on a DocSet.

        Args:
            field: Field to aggregate based on.
            unique_field: Determines what makes a unique document.
            **kwargs

        Returns:
            A DocSet with "properties.key" (unique values of document field)
            and "properties.count" (frequency counts for unique values).
        """
        from sycamore.transforms import GroupByCount
        from sycamore.transforms import DatasetScan

        dataset = GroupByCount(self.plan, field, unique_field).execute(**kwargs)
        return DocSet(self.context, DatasetScan(dataset))

    def llm_query(self, query_agent: LLMTextQueryAgent, **kwargs) -> "DocSet":
        """
        Executes an LLM Query on a specified field (element or document), and returns the response

        Example:
            .. code-block:: python

                prompt="Tell me the important numbers from this element"
                llm_query_agent = LLMElementTextSummarizer(prompt=prompt)

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=UnstructuredPdfPartitioner())
                    .llm_query(query_agent=llm_query_agent)
        """
        from sycamore.transforms import LLMQuery

        queries = LLMQuery(self.plan, query_agent=query_agent, **kwargs)
        return DocSet(self.context, queries)

    def top_k(
        self,
        llm: LLM,
        field: str,
        k: Optional[int],
        descending: bool = True,
        llm_cluster: bool = False,
        unique_field: Optional[str] = None,
        llm_cluster_description: Optional[str] = None,
        **kwargs,
    ) -> "DocSet":
        """
        Determines the top k occurrences for a document field.

        Args:
            llm: LLM client.
            field: Field to determine top k occurrences of.
            k: Number of top occurrences. If k is not specified, all occurences are returned.
            llm_cluster_description: Description of operation purpose.  E.g. Find most common cities
            descending: Indicates whether to return most or least frequent occurrences.
            llm_cluster: Indicates whether an LLM should be used to normalize values of document field.
            unique_field: Determines what makes a unique document.
            **kwargs

        Returns:
            A DocSet with "properties.key" (unique values of document field)
            and "properties.count" (frequency counts for unique values) which is
            sorted based on descending and contains k records.
        """

        docset = self

        if llm_cluster:
            if llm_cluster_description is None:
                raise Exception("Description of groups must be provided to form clusters.")
            docset = docset.llm_cluster_entity(llm, llm_cluster_description, field)
            field = "properties._autogen_ClusterAssignment"

        docset = docset.groupby_count(field, unique_field, **kwargs)

        docset = docset.sort(descending, "properties.count", 0)
        if k is not None:
            docset = docset.limit(k)
        return docset

    def llm_cluster_entity(self, llm: LLM, description: str, field: str) -> "DocSet":
        """
        Normalizes a particular field of a DocSet. Identifies and assigns each document to a "group".

        Args:
            llm: LLM client.
            description: Description of purpose of this operation.
            field: Field to make/assign groups based on.

        Returns:
            A DocSet with an additional field "properties._autogen_ClusterAssignment" that contains
            the assigned group.
        """

        docset = self
        text = ", ".join([doc.field_to_value(field) for doc in docset.take_all()])

        # sets message
        messages = LlmClusterEntityFormGroupsMessagesPrompt(
            field=field, description=description, text=text
        ).get_messages_dict()

        prompt_kwargs = {"messages": messages}

        # call to LLM
        completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

        groups = extract_json(completion)

        assert isinstance(groups, dict)

        # sets message
        messagesForExtract = LlmClusterEntityAssignGroupsMessagesPrompt(
            field=field, groups=groups["groups"]
        ).get_messages_dict()

        entity_extractor = OpenAIEntityExtractor(
            entity_name="_autogen_ClusterAssignment",
            llm=llm,
            use_elements=False,
            prompt=messagesForExtract,
            field=field,
        )
        docset = docset.extract_entity(entity_extractor=entity_extractor)

        # LLM response
        return docset

    def field_in(self, docset2: "DocSet", field1: str, field2: str) -> "DocSet":
        """
        Joins two docsets based on specified fields; docset (self) filtered based on values of docset2.

        SQL Equivalent: SELECT * FROM docset1 WHERE field1 IN (SELECT field2 FROM docset2);

        Args:
            docset2: DocSet to filter.
            field1: Field in docset1 to filter based on.
            field2: Field in docset2 to filter.

        Returns:
            A left semi-join between docset (self) and docset2.
        """

        from sycamore import Execution

        def make_filter_fn_join(field: str, join_set: set) -> Callable[[Document], bool]:
            def filter_fn_join(doc: Document) -> bool:
                value = doc.field_to_value(field)
                return value in join_set

            return filter_fn_join

        execution = Execution(docset2.context, docset2.plan)
        dataset = execution.execute(docset2.plan)

        # identifies unique values of field1 in docset (self)
        unique_vals = set()
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if isinstance(doc, MetadataDocument):
                continue
            value = doc.field_to_value(field2)
            unique_vals.add(value)

        # filters docset2 based on matches of field2 with unique values
        filter_fn_join = make_filter_fn_join(field1, unique_vals)
        joined_docset = self.filter(lambda doc: filter_fn_join(doc))

        return joined_docset

    @property
    def write(self) -> DocSetWriter:
        """
        Exposes an interface for writing a DocSet to OpenSearch or other external storage.
        See :class:`~writer.DocSetWriter` for more information about writers and their arguments.

        Example:
             The following example shows reading a DocSet from a collection of PDFs, partitioning
             it using the ``ArynPartitioner``, and then writing it to a new OpenSearch index.

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
                    .partition(partitioner=ArynPartitioner())

                pdf.write.opensearch(
                     os_client_args=os_client_args,
                     index_name="my_index",
                     index_settings=index_settings)
        """
        return DocSetWriter(self.context, self.plan)

    def materialize(self, path: Optional[Union[Path, str, dict]] = None) -> "DocSet":
        """
        Guarantees reliable execution up to this point, allows for
        follow on execution based on the checkpoint if the checkpoint is named.

        path: a Path or string represents the "directory" for the materialized elements. The filesystem
              and naming convention will be inferred.  The dictionary allowes finer control, and supports
              { root=Path|str, fs=pyarrow.fs, name=lambda Document -> str, clean=True } where the root is required
        """

        from sycamore.lineage import Materialize

        return DocSet(self.context, Materialize(self.plan, self.context, path=path))

    def execute(self, **kwargs) -> None:
        """
        Execute the pipeline, discard the results. Useful for side effects.
        """

        from sycamore.executor import Execution

        execution = Execution(self.context, self.plan)
        for doc in execution.execute_iter(self.plan, **kwargs):
            pass
