# Please don't add more to these; every one slows down importing transforms if they are unused.
# Worse importing any transform forces importing all of them.
from sycamore.transforms.embed import Embed, Embedder
from sycamore.transforms.basics import Limit, Filter
from sycamore.transforms.extract_document_structure import DocumentStructure, ExtractDocumentStructure
from sycamore.transforms.extract_entity import ExtractEntity, EntityExtractor
from sycamore.transforms.explode import Explode
from sycamore.transforms.map import Map, FlatMap, MapBatch
from sycamore.transforms.partition import Partition, Partitioner

# from sycamore.transforms.extract_table import TableExtractor
from sycamore.transforms.regex_replace import COALESCE_WHITESPACE, RegexReplace
from sycamore.transforms.similarity import ScoreSimilarity
from sycamore.transforms.sketcher import Sketcher, SketchUniquify, SketchDebug
from sycamore.transforms.spread_properties import SpreadProperties
from sycamore.transforms.assign_doc_properties import AssignDocProperties
from sycamore.transforms.summarize import Summarize
from sycamore.transforms.bbox_merge import (
    SortByPageBbox,
    MarkDropHeaderFooter,
    MarkBreakByColumn,
)

from sycamore.transforms.extract_table_properties import ExtractTableProperties

from sycamore.transforms.standardizer import (
    USStateStandardizer,
    Standardizer,
    StandardizeProperty,
    DateTimeStandardizer,
)
from sycamore.transforms.mark_misc import (
    MarkDropTiny,
    MarkBreakPage,
    MarkBreakByTokens,
)
from sycamore.transforms.merge_elements import Merge

from sycamore.transforms.random_sample import RandomSample
from sycamore.transforms.split_elements import SplitElements
from sycamore.transforms.query import Query
from sycamore.transforms.term_frequency import TermFrequency
from sycamore.transforms.sort import Sort
from sycamore.transforms.llm_query import LLMQuery
from sycamore.transforms.groupby_count import GroupByCount
from sycamore.transforms.dataset_scan import DatasetScan


# commented out bits can be removed after 2025-08-01; they are here to help people
# find where things should be imported from
__all__ = [
    "COALESCE_WHITESPACE",
    "Explode",
    "FlatMap",
    "Limit",
    "Map",
    "MapBatch",
    "Partitioner",
    "Embed",
    "Embedder",
    "Partition",
    "DocumentStructure",
    "ExtractDocumentStructure",
    "ExtractEntity",
    "EntityExtractor",
    #    "TableExtractor", # sycamore.transforms.extract_table
    "SpreadProperties",
    "RegexReplace",
    "Sketcher",
    "SketchUniquify",
    "SketchDebug",
    "Summarize",
    "SortByPageBbox",
    "MarkBreakByColumn",
    "MarkBreakPage",
    "MarkBreakByTokens",
    "MarkDropTiny",
    "MarkDropHeaderFooter",
    "Filter",
    "Merge",
    #    "ExtractSchema",  # sycamore.transforms.extract_schema
    #    "ExtractBatchSchema",  # sycamore.transforms.extract_schema
    #    "SchemaExtractor",  # sycamore.transforms.extract_schema
    #    "PropertyExtractor",  # sycamore.transforms.extract_schema
    "RandomSample",
    "SplitElements",
    "Query",
    "TermFrequency",
    "Sort",
    "LLMQuery",
    "AssignDocProperties",
    "USStateStandardizer",
    "Standardizer",
    "DateTimeStandardizer",
    "StandardizeProperty",
    "ExtractTableProperties",
    "GroupByCount",
    "DatasetScan",
    "ScoreSimilarity",
]
