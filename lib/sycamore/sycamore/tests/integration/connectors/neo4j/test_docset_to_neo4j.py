from typing import Optional
import pytest
import sycamore
from sycamore.llms.llms import LLM
from sycamore.reader import DocSetReader
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.extract_graph import GraphMetadata, MetadataExtractor, GraphEntity, EntityExtractor
from sycamore.data import HierarchicalDocument, Document

from sycamore.transforms.partition import ArynPartitioner


########## THESE FUNCTIONS NEED A HOME
@pytest.mark.skip
def restructure_doc(doc: Document) -> HierarchicalDocument:
    doc = HierarchicalDocument(doc.data)     
    return doc

@pytest.mark.skip
def children_to_section(doc: HierarchicalDocument) -> HierarchicalDocument:
    import uuid
    
    #if the first element is not a section header, insert generic placeholder
    if(len(doc.children) > 0 and doc.children[0]["type"] != 'Section-header'):
        initial_page = HierarchicalDocument({'type': 'Section-header', 'bbox': (0, 0, 0, 0), 'properties': {'score': 1, 'page_number': 1}, 'text_representation': 'Front Page', 'binary_representation': b'Front Page'})
        doc.children.insert(0,initial_page) # O(n) insert :( we should use deque for everything

    if 'relationships' not in doc.data:
        doc.data['relationships'] = {}
    if 'label' not in doc.data:
        doc.data['label'] = 'DOCUMENT'

    sections = []

    section: HierarchicalDocument = None
    element: HierarchicalDocument = None
    for child in doc.children:
        if 'relationships' not in child.data:
            child.data['relationships'] = {}
        if child.type == 'Section-header' and 'text_representation' in child.data and len(child.data["text_representation"]) > 0:
            if section != None:
                next = {'TYPE':'NEXT',
                        'properties': {},
                        'START_ID': section.doc_id,
                        'START_LABEL':'SECTION',
                        'END_ID':child.doc_id,
                        'END_LABEL':'SECTION',
                        }
                child.data['relationships'][str(uuid.uuid4())] = next
                element = None
            rel = {'TYPE':'SECTION_OF',
                        'properties': {},
                        'START_ID': child.doc_id,
                        'START_LABEL': 'SECTION',
                        'END_ID':doc.doc_id,
                        'END_LABEL': 'DOCUMENT'
                        }
            child.data['relationships'][str(uuid.uuid4())] = rel
            child.data['label'] = 'SECTION'
            section = child
            sections.append(section)
        else:
            if element != None:
                next = {'TYPE':'NEXT',
                        'properties': {},
                        'START_ID': element.doc_id,
                        'START_LABEL': 'ELEMENT',
                        'END_ID':child.doc_id,
                        'END_LABEL': 'ELEMENT'
                        }
                child.data['relationships'][str(uuid.uuid4())] = next
            rel = {'TYPE':'PART_OF',
                        'properties': {},
                        'START_ID': child.doc_id,
                        'START_LABEL': 'ELEMENT',
                        'END_ID': section.doc_id,
                        'END_LABEL': 'SECTION'
                        }
            child.data['relationships'][str(uuid.uuid4())] = rel
            child.data['label'] = 'ELEMENT'
            element = child
            section.data["children"].append(element)

    doc.children = sections
    return doc
##########


    
def test_docset_to_neo4j():
    path = str(TEST_DIR / "resources/data/pdfs/Ray.pdf")
    context = sycamore.init()
    #URI = "neo4j://localhost:7687"
    #AUTH = ("neo4j", "koala-stereo-comedy-spray-figure-6974")

    ds = (
        context.read.binary(path, binary_format="pdf")
        .partition(partitioner=ArynPartitioner(extract_table_structure=True, use_ocr=True, extract_images=True))
        .map(restructure_doc)
        .map(children_to_section)
        .explode()
    )

    #ds.write.neo4j(uri=URI,auth=AUTH,database="neo4j",import_dir="/neo4j/import")

