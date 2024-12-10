from collections.abc import Mapping
import logging
from pathlib import Path
import pprint
import sys
from typing import Callable, Optional, Any, Iterable, Type, Union, TYPE_CHECKING
import re

from sycamore.context import Context, context_params, OperationTypes
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
from sycamore.transforms import DocumentStructure, Sort
from sycamore.transforms.extract_entity import EntityExtractor, OpenAIEntityExtractor
from sycamore.transforms.extract_graph_entities import GraphEntityExtractor
from sycamore.transforms.extract_graph_relationships import GraphRelationshipExtractor
from sycamore.transforms.extract_schema import SchemaExtractor, PropertyExtractor
from sycamore.transforms.partition import Partitioner
from sycamore.transforms.similarity import SimilarityScorer
from sycamore.transforms.resolve_graph_entities import EntityResolver, ResolveEntities
from sycamore.transforms.summarize import Summarizer
from sycamore.transforms.llm_query import LLMTextQueryAgent
from sycamore.transforms.extract_table import TableExtractor
from sycamore.transforms.merge_elements import ElementMerger
from sycamore.utils.extract_json import extract_json
from sycamore.transforms.query import QueryExecutor, Query
from sycamore.materialize_config import MaterializeSourceMode

if TYPE_CHECKING:
    from sycamore.writer import DocSetWriter

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

        count = 0
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
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
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
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

        ret = []
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
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
        has more than `limit` Documents.

        Args:
            limit: The number of Documents above which this method will raise an error.
        """
        from sycamore import Execution

        docs = []
        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
            if include_metadata or not isinstance(doc, MetadataDocument):
                docs.append(doc)

            if limit is not None and len(docs) > limit:
                raise ValueError(f"docset exceeded limit of {limit} docs")

        return docs

    def take_streaming(self, include_metadata: bool = False, **kwargs) -> Iterable[Document]:
        """
        Returns a stream of all rows in this DocSet.

        Args:
            include_metadata: False [default] will filter out all MetadataDocuments from the result.
        """
        from sycamore import Execution

        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
            if include_metadata or not isinstance(doc, MetadataDocument):
                yield doc

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

            augmentor = UDFTextAugmentor(
                lambda doc: f"This pertains to the part {doc.properties['part_name']}.\n{doc.text_representation}"
            )
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

    def extract_document_structure(self, structure: DocumentStructure, **kwargs):
        """
        Represents documents as Hierarchical documents organized by their structure.

        Args:
            structure: A instance of DocumentStructure which determines how documents are organized

        Example:
            .. code-block:: python

                context = sycamore.init()
                pdf_docset = context.read.binary(paths, binary_format="pdf")
                    .partition(partitioner=ArynPartitioner())
                    .extract_document_structure(structure=StructureBySection)
                    .explode()

        """
        from sycamore.transforms.extract_document_structure import ExtractDocumentStructure

        document_structure = ExtractDocumentStructure(self.plan, structure=structure, **kwargs)
        return DocSet(self.context, document_structure)

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

        entities = ExtractEntity(self.plan, context=self.context, entity_extractor=entity_extractor, **kwargs)
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

    def extract_graph_entities(self, extractors: list[GraphEntityExtractor] = [], **kwargs) -> "DocSet":
        """
        Extracts entites from document children. Entities are stored as nodes within each child of
        a document.

        Args:
            extractors: A list of GraphEntityExtractor objects which determines how entities are extracted

        Example:
            .. code-block:: python
                from sycamore.transforms.extract_graph_entities import EntityExtractor
                from pydantic import BaseModel

                llm = OpenAI(OpenAIModels.GPT_4O_MINI.value)

                class CEO(BaseModel):
                    name: str

                class Company(BaseModel):
                    name: str

                ds = (
                    context.read.binary(paths, binary_format="pdf")
                    .partition(...)
                    .extract_document_structure(...)
                    .extract_graph_entities(extractors=[EntityExtractor(llm=llm, entities=[CEO, Company])])
                    .resolve_graph_entities(...)
                    .explode()
                )

                ds.write.neo4j(...)
        """
        from sycamore.transforms.extract_graph_entities import ExtractEntities

        entities = self.plan
        for extractor in extractors:
            entities = ExtractEntities(entities, extractor)

        return DocSet(self.context, entities)

    def extract_graph_relationships(self, extractors: list[GraphRelationshipExtractor] = [], **kwargs) -> "DocSet":
        """
        Extracts relationships from document children. Relationships are stored within the nodes they reference
        within each child of a document.

        Args:
            extractors: A list of GraphEntityExtractor objects which determines how relationships are extracted

        Example:
            .. code-block:: python
                from sycamore.transforms.extract_graph_entities import EntityExtractor
                from sycamore.transforms.extract_graph_relationships import RelationshipExtractor
                from pydantic import BaseModel

                llm = OpenAI(OpenAIModels.GPT_4O_MINI.value)

                class CEO(BaseModel):
                    name: str

                class Company(BaseModel):
                    name: str

                class WORKS_AT(BaseModel):
                    start: CEO
                    end: Company

                ds = (
                    context.read.binary(paths, binary_format="pdf")
                    .partition(...)
                    .extract_document_structure(...)
                    .extract_graph_entities(extractors=[EntityExtractor(llm=llm, entities=[CEO, Company])])
                    .extract_graph_relationships(extractors=[RelationshipExtractor(llm=llm, relationships=[WORKS_AT])])
                    .resolve_graph_entities(...)
                    .explode()
                )
                ds.write.neo4j(...)
        """
        from sycamore.transforms.extract_graph_relationships import ExtractRelationships

        relationships = self.plan
        for extractor in extractors:
            relationships = ExtractRelationships(relationships, extractor)

        return DocSet(self.context, relationships)

    def resolve_graph_entities(
        self, resolvers: list[EntityResolver] = [], resolve_duplicates=True, **kwargs
    ) -> "DocSet":
        """
        Resolves graph entities across documents so that duplicate entities can be resolved
        to the same entity based off criteria of EntityResolver objects.

        Args:
            resolvers: A list of EntityResolvers that are used to determine what entities are duplicates
            resolve_duplicates: If exact duplicate entities and relationships should be merged. Defaults to true

        Example:
            .. code-block:: python
                ds = (
                    context.read.binary(paths, binary_format="pdf")
                    .partition(...)
                    .extract_document_structure(...)
                    .extract_graph_entities(...)
                    .extract_graph_relationships(...)
                    .resolve_graph_entities(resolvers=[], resolve_duplicates=False)
                    .explode()
                )
                ds.write.neo4j(...)
        """
        from sycamore.transforms.resolve_graph_entities import CleanTempNodes

        class Wrapper(Node):
            def __init__(self, dataset):
                super().__init__(children=[])
                self._ds = dataset

            def execute(self, **kwargs):
                return self._ds

        entity_resolver = ResolveEntities(resolvers=resolvers, resolve_duplicates=resolve_duplicates)
        entities = entity_resolver.resolve(self)  # resolve entities
        entities_clean = CleanTempNodes(Wrapper(entities))  # cleanup temp objects
        return DocSet(self.context, entities_clean)

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

        |    SortByPageBbox
        |    MarkDropTiny minimum=2
        |    MarkDropHeaderFooter top=0.05 bottom=0.05
        |    MarkBreakPage
        |    MarkBreakByColumn
        |    MarkBreakByTokens limit=512

        Meant to work in concert with MarkedMerger.

        Use this method like so:

        .. code-block:: python

            context = sycamore.init()
            token_limit = 512
            paths = ["path/to/pdf1.pdf", "path/to/pdf2.pdf"]

            (context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=ArynPartitioner())
                .mark_bbox_preset(tokenizer, token_limit)
                .merge(merger=MarkedMerger())
                .split_elements(tokenizer, token_limit)
                .show())

        If you want to compose your own marking, note that ``docset.mark_bbox_preset(...)`` is equivalent to:

        .. code-block:: python

            (docset.transform(SortByPageBbox)
                .transform(MarkDropTiny, minimum=2)
                .transform(MarkDropHeaderFooter, top=0.05, bottom=0.05)
                .transform(MarkBreakPage)
                .transform(MarkBreakByColumn)
                .transform(MarkBreakByTokens, tokenizer=tokenizer, limit=token_limit))
        """
        from sycamore.transforms.mark_misc import MarkBboxPreset

        preset = MarkBboxPreset(self.plan, tokenizer, token_limit, **kwargs)
        return DocSet(self.context, preset)

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

    def markdown(self, **kwargs) -> "DocSet":
        """
        Modifies Document to have a single Element containing the Markdown
        representation of all the original elements.

        Example:
            .. code-block:: python

               context = sycamore.init()
               ds = context.read.binary(paths, binary_format="pdf")
                   .partition(partitioner=ArynPartitioner())
                   .markdown()
                   .explode()
        """
        from sycamore.transforms.markdown import Markdown

        plan = Markdown(self.plan, **kwargs)
        return DocSet(self.context, plan)

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
                   .regex_replace([(r"\\d+", "1313"), (r"old", "new")])
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

        See the :class:`~sycamore.transforms.map.Map` documentation for advanced features.
        """
        from sycamore.transforms import Map

        mapping = Map(self.plan, f=f, **resource_args)
        return DocSet(self.context, mapping)

    def flat_map(self, f: Callable[[Document], list[Document]], **resource_args) -> "DocSet":
        """
        Applies the FlatMap transformation on the Docset.

        Args:
            f: The function to apply to each document.

        See the :class:`~sycamore.transforms.map.FlatMap` documentation for advanced features.

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

    def filter(self, f: Callable[[Document], bool], **kwargs) -> "DocSet":
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

        filtered = Filter(self.plan, f=f, **kwargs)
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

    @context_params(OperationTypes.BINARY_CLASSIFIER)
    @context_params(OperationTypes.TEXT_SIMILARITY)
    def llm_filter(
        self,
        llm: LLM,
        new_field: str,
        prompt: Union[list[dict], str],
        field: str = "text_representation",
        threshold: int = 3,
        keep_none: bool = False,
        use_elements: bool = False,
        similarity_query: Optional[str] = None,
        similarity_scorer: Optional[SimilarityScorer] = None,
        **resource_args,
    ) -> "DocSet":
        """
        Filters DocSet to only keep documents that score (determined by LLM) greater
        than or equal to the inputted threshold value.

        Args:
            llm: LLM to use.
            new_field: The field that will be added to the DocSet with the outputs.
            prompt: LLM prompt.
            field: Document field to filter based on.
            threshold:  If the value of the computed result is an integer value greater than or equal to this threshold,
                        the document will be kept.
            keep_none:  keep records with a None value for the provided field to filter on.
                        Warning: using this might hide data corruption issues.
            use_elements: use contents of a document's elements to filter as opposed to document level contents.
            similarity_query: query string to compute similarity against. Also requires a 'similarity_scorer'.
            similarity_scorer: scorer used to generate similarity scores used in element sorting.
                        Also requires a 'similarity_query'.
            **resource_args

        Returns:
            A filtered DocSet.
        """
        entity_extractor = OpenAIEntityExtractor(
            entity_name=new_field, llm=llm, use_elements=False, prompt=prompt, field=field
        )

        def threshold_filter(doc: Document, threshold) -> bool:
            if not use_elements:
                if doc.field_to_value(field) is None:
                    return keep_none
                doc = entity_extractor.extract_entity(doc)
                # todo: move data extraction and validation to entity extractor
                return int(re.findall(r"\d+", doc.properties[new_field])[0]) >= threshold

            if similarity_query is not None:
                assert similarity_scorer is not None, "Similarity sorting requires a scorer"
                score_property_name = f"{field}_similarity_score"
                doc = similarity_scorer.generate_similarity_scores(
                    doc_batch=[doc], query=similarity_query, score_property_name=score_property_name
                )[0]
                doc.elements.sort(key=lambda e: e.properties.get(score_property_name, float("-inf")), reverse=True)
            evaluated_elements = 0
            for element in doc.elements:
                e_doc = Document(element.data)
                if e_doc.field_to_value(field) is None:
                    continue
                e_doc = entity_extractor.extract_entity(e_doc)
                element.properties[new_field] = e_doc.properties[new_field]

                # todo: move data extraction and validation to entity extractor
                score = int(re.findall(r"\d+", element.properties[new_field])[0])
                # we're storing the element_index of the element that provides the highest match score for a document.
                doc_source_field_name = f"{new_field}_source_element_index"
                if score >= doc.get(doc_source_field_name, 0):
                    doc.properties[f"{new_field}"] = score
                    doc.properties[f"{new_field}_source_element_index"] = element.element_index
                if score >= threshold:
                    return True
                evaluated_elements += 1
            if evaluated_elements == 0:  # no elements found for property
                return keep_none
            return False

        docset = self.filter(lambda doc: threshold_filter(doc, threshold), **resource_args)

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

        See the :class:`~sycamore.transforms.map.MapBatch` documentation for advanced features.

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

    @context_params(OperationTypes.TEXT_SIMILARITY)
    def rerank(
        self,
        similarity_scorer: SimilarityScorer,
        query: str,
        score_property_name: str = "_rerank_score",
        limit: Optional[int] = None,
    ) -> "DocSet":
        """
        Sort a DocSet given a scoring class.

        Args:
            similarity_scorer: An instance of an SimilarityScorer class that executes the scoring function.
            query: The query string to compute similarity against.
            score_property_name: The name of the key where the score will be stored in document.properties
            limit: Limit scoring and sorting to fixed size.
        """
        from sycamore.transforms import ScoreSimilarity, Limit

        if limit:
            plan = Limit(self.plan, limit)
        else:
            plan = self.plan
        similarity_scored = ScoreSimilarity(
            plan, similarity_scorer=similarity_scorer, query=query, score_property_name=score_property_name
        )
        return DocSet(
            self.context,
            Sort(
                similarity_scored, descending=True, field=f"properties.{score_property_name}", default_val=float("-inf")
            ),
        )

    def sort(self, descending: bool, field: str, default_val: Optional[Any] = None) -> "DocSet":
        """
        Sort DocSet by specified field.

        Args:
            descending: Whether or not to sort in descending order (first to last).
            field: Document field in relation to Document using dotted notation, e.g. properties.filetype
            default_val: Default value to use if field does not exist in Document
        """
        from sycamore.transforms.sort import Sort, DropIfMissingField

        plan = self.plan
        if default_val is None:
            import logging

            logging.warning(
                "Default value is none. Adding explicit filter step to drop documents missing the key."
                " This includes any metadata.documents."
            )
            plan = DropIfMissingField(plan, field)
        return DocSet(self.context, Sort(plan, descending, field, default_val))

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

        Args:
            prompt: A prompt to be passed into the underlying LLM execution engine
            llm: The LLM Client to be used here. It is defined as an instance of the LLM class in Sycamore.
            output_property: (Optional, default="llm_response") The output property of the document or element to add
                results in.
            format_kwargs: (Optional, default="None") If passed in, details the formatting details that must be
                passed into the underlying Jinja Sandbox.
            number_of_elements: (Optional, default="None") When "per_element" is true, limits the number of
                elements to add an "output_property". Otherwise, the response is added to the
                entire document using a limited prefix subset of the elements.
            llm_kwargs: (Optional) LLM keyword argument for the underlying execution engine
            per_element: (Optional, default="{}") Keyword arguments to be passed into the underlying LLM execution
                engine.
            element_type: (Optional) Parameter to only execute the LLM query on a particular element type. If not
                specified, the query will be executed on all elements.
        """
        from sycamore.transforms import LLMQuery

        queries = LLMQuery(self.plan, query_agent=query_agent, **kwargs)
        return DocSet(self.context, queries)

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    def top_k(
        self,
        llm: Optional[LLM],
        field: str,
        k: Optional[int],
        descending: bool = True,
        llm_cluster: bool = False,
        unique_field: Optional[str] = None,
        llm_cluster_instruction: Optional[str] = None,
        **kwargs,
    ) -> "DocSet":
        """
        Determines the top k occurrences for a document field.

        Args:
            llm: LLM client.
            field: Field to determine top k occurrences of.
            k: Number of top occurrences. If k is not specified, all occurences are returned.
            llm_cluster_instruction: Instruction of operation purpose.  E.g. Find most common cities
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
            if llm_cluster_instruction is None:
                raise Exception("Description of groups must be provided to form clusters.")
            docset = docset.llm_cluster_entity(llm, llm_cluster_instruction, field)
            field = "properties._autogen_ClusterAssignment"

        docset = docset.groupby_count(field, unique_field, **kwargs)

        docset = docset.sort(descending, "properties.count", 0)
        if k is not None:
            docset = docset.limit(k)
        return docset

    @context_params(OperationTypes.INFORMATION_EXTRACTOR)
    def llm_cluster_entity(self, llm: LLM, instruction: str, field: str, **kwargs) -> "DocSet":
        """
        Normalizes a particular field of a DocSet. Identifies and assigns each document to a "group".

        Args:
            llm: LLM client.
            instruction: Instruction about groups to form, e.g. 'Form groups for different types of food'
            field: Field to make/assign groups based on, e.g. 'properties.entity.food'

        Returns:
            A DocSet with an additional field "properties._autogen_ClusterAssignment" that contains
            the assigned group. For example, if "properties.entity.food" has values 'banana', 'milk',
            'yogurt', 'chocolate', 'orange', "properties._autogen_ClusterAssignment" would contain
            values like 'fruit', 'dairy', and 'dessert'.
        """

        docset = self
        # Not all documents will have a value for the given field, so we filter those out.
        field_values = [doc.field_to_value(field) for doc in docset.take_all()]
        text = ", ".join([str(v) for v in field_values if v is not None])

        # sets message
        messages = LlmClusterEntityFormGroupsMessagesPrompt(
            field=field, instruction=instruction, text=text
        ).as_messages()

        prompt_kwargs = {"messages": messages}

        # call to LLM
        completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

        groups = extract_json(completion)

        assert isinstance(groups, dict)

        # sets message
        messagesForExtract = LlmClusterEntityAssignGroupsMessagesPrompt(
            field=field, groups=groups["groups"]
        ).as_messages()

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

    def field_in(self, docset2: "DocSet", field1: str, field2: str, **kwargs) -> "DocSet":
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

        # identifies unique values of field1 in docset (self)
        unique_vals = set()
        for doc in Execution(docset2.context).execute_iter(docset2.plan, **kwargs):
            if isinstance(doc, MetadataDocument):
                continue
            value = doc.field_to_value(field2)
            unique_vals.add(value)

        # filters docset2 based on matches of field2 with unique values
        filter_fn_join = make_filter_fn_join(field1, unique_vals)
        joined_docset = self.filter(lambda doc: filter_fn_join(doc))

        return joined_docset

    @property
    def write(self) -> "DocSetWriter":
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
        from sycamore.writer import DocSetWriter

        return DocSetWriter(self.context, self.plan)

    def materialize(
        self,
        path: Optional[Union[Path, str, dict]] = None,
        source_mode: MaterializeSourceMode = MaterializeSourceMode.RECOMPUTE,
    ) -> "DocSet":
        """
        The `materialize` transform writes out documents up to that point, marks the
        materialized path as successful if execution is successful, and allows for reading from the
        materialized data as a source. This transform is helpful if you are using show and take()
        as part of a notebook to incrementally inspect output. You can use `materialize` to avoid
        re-computation.

        path: a Path or string represents the "directory" for the materialized elements. The filesystem
              and naming convention will be inferred.  The dictionary variant allowes finer control, and supports
              { root=Path|str, fs=pyarrow.fs, name=lambda Document -> str, clean=True,
                tobin=Document.serialize()}
              root is required

        source_mode: how this materialize step should be used as an input:
           RECOMPUTE: (default) the transform does not act as a source, previous transforms
             will be recomputed.
           USE_STORED: If the materialize has successfully run to completion, or if the
             materialize step has no prior step, use the stored contents of the directory as the
             inputs.  No previous transform will be computed.
             WARNING: If you change the input files or any of the steps before the
             materialize step, you need to use clear_materialize() or change the source_mode
             to force re-execution.

           Note: you can write the source mode as MaterializeSourceMode.SOMETHING after importing
           MaterializeSourceMode, or as sycamore.MATERIALIZE_SOMETHING after importing sycamore.

        """

        from sycamore.materialize import Materialize

        return DocSet(self.context, Materialize(self.plan, self.context, path=path, source_mode=source_mode))

    def clear_materialize(self, path: Optional[Union[Path, str]] = None, *, clear_non_local=False) -> None:
        """
        Deletes all of the materialized files referenced by the docset.

        path will use PurePath.match to check if the specified path matches against
        the directory used for each materialize transform. Only matching directories
        will be cleared.

        Set clear_non_local=True to clear non-local filesystems. Note filesystems like
        NFS/CIFS will count as local.  pyarrow.fs.SubTreeFileSystem is treated as non_local.
        """

        from sycamore.materialize import clear_materialize

        clear_materialize(self.plan, path=path, clear_non_local=clear_non_local)

    def execute(self, **kwargs) -> None:
        """
        Execute the pipeline, discard the results. Useful for side effects.
        """

        from sycamore.executor import Execution

        for doc in Execution(self.context).execute_iter(self.plan, **kwargs):
            pass
