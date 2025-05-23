import ast
import pickle
from collections import Counter
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def compute_distances(list_a, list_b, num_pairs=10):
    distances = []
    for _ in range(num_pairs):
        vec1 = random.choice(list_a)
        vec2 = random.choice(list_b)
        dist = cosine_distances([np.array(vec1)], [np.array(vec2)])[0][0]
        distances.append(dist)
    return distances

def get_elem_vect(doc, e_i):
    """
    Find the element in the document with the given index.
    """
    #print(f"e_i: {e_i}")

    for i, elem in enumerate(doc.elements):
        #print(elem)
        if elem["properties"]["_autogen_LLMFilterOutput_source_indices"] == e_i:
            #print(f'TYpe {type(elem["embedding"])}')
            return elem["embedding"]

    return None

def get_elem_id (doc, e_i):
    """
    Find the element in the document with the given index.
    """
    #print(f"e_i: {e_i}")

    for i, elem in enumerate(doc.elements):
        #print(elem)
        if elem["properties"]["_autogen_LLMFilterOutput_source_indices"] == e_i:
            #print(f'TYpe {type(elem["embedding"])}')
            return elem["properties"]["element_id"]+doc["doc_id"]
    print(f"{e_i} not found in {doc}")
    return None

def get_matching_elem_vect(doc):
    # HACK: this assumes batch size is 1
    last_batch_indx = int(doc["properties"]["_autogen_LLMFilterOutput_i"])
    batches = doc["properties"]["_autogen_LLMFilterOutput_batches"]
    return get_elem_vect(doc, batches[last_batch_indx])


def get_match_nomatch_elems(doc):
    """
    Find the elements in the document thta mathced and did not match the llm_filter
    """
    e_match = []
    e_nomatch = []
    #print(f"e_i: {e_i}")
    #print(f'Doc is {doc}')
    last_batch_indx = int(doc["properties"]["_autogen_LLMFilterOutput_i"])
    # all elems up to last_batch_indx did not match
    batches = doc["properties"]["_autogen_LLMFilterOutput_batches"]
    for batch in batches[:last_batch_indx]:
        e_nomatch.append(get_elem_vect(doc, batch))
    e_match.append(get_elem_vect(doc, batches[last_batch_indx]))

    return e_match, e_nomatch

def get_match_elem_ids(doc):
    """
    Find the element in the document with the given index.
    """
    #print(f"e_i: {e_i}")
    batches = doc["properties"]["_autogen_LLMFilterOutput_batches"]
    last_batch_indx = int(doc["properties"]["_autogen_LLMFilterOutput_i"])
    return get_elem_id(doc, batches[last_batch_indx])

def get_match_nomatch_elems_full(doc):
    """
    Find the elements in the document thta mathced and did not match the llm_filter
    """
    e_match = []
    e_nomatch = []
    #print(f"e_i: {e_i}")
    #print(f'Doc is {doc}')
    last_batch_indx = int(doc["properties"]["_autogen_LLMFilterOutput_i"])
    # all elems up to last_batch_indx did not match
    batches = doc["properties"]["_autogen_LLMFilterOutput_batches"]
    for batch in batches[:last_batch_indx]:
        e_nomatch.append(get_elem_vect(doc, batch))
    e_match.append(get_elem_vect(doc, batches[last_batch_indx]))

    return e_match, e_nomatch

def extract_match_nomatch(data):
    elems_processed = 0
    no_batches = 0
    e_match = []
    e_nomatch = []
    for d in data:
        if "_autogen_LLMFilterOutput_batches" in d["properties"]:
            match, no_match = get_match_nomatch_elems(d)
            batches = d["properties"]["_autogen_LLMFilterOutput_batches"]
            last_batch_indx = int(d["properties"]["_autogen_LLMFilterOutput_i"])
            elems_processed += len(batches[:last_batch_indx])+1
            e_match.extend(match)
            e_nomatch.extend(no_match)
        else:
            no_batches += 1
    # Print the result
    print(f'Documents processed: {len(data)}')
    print(f'Elements processed via llm_filter: {elems_processed}')
    print(f'Docs with no batches: {no_batches}')
    print(f'Elements matched: {len(e_match)}')
    print(f'Elements not matched: {len(e_nomatch)}')
    return e_match, e_nomatch, no_batches, elems_processed