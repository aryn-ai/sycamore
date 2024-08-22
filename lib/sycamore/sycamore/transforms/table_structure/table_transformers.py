"""
This file is derived from the TableTransformer project (https://github.com/microsoft/table-transformer), which
is copyright (C) 2021 Microsoft Corporation licensed under the MIT license.

Modifications Copyright Aryn Inc.
"""

from collections import OrderedDict, defaultdict
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from sycamore.data.table import Table, TableCell
from sycamore.data import BoundingBox


# From https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def apply_threshold(objects, threshold):
    """
    Filter out objects below a certain score.
    """
    return [obj for obj in objects if obj["score"] >= threshold]


def apply_class_thresholds(bboxes, labels, scores, class_names, class_thresholds):
    """
    Filter out bounding boxes whose confidence is below the confidence threshold for
    its associated class label.
    """
    # Apply class-specific thresholds
    indices_above_threshold = [
        idx for idx, (score, label) in enumerate(zip(scores, labels)) if score >= class_thresholds[class_names[label]]
    ]
    bboxes = [bboxes[idx] for idx in indices_above_threshold]
    scores = [scores[idx] for idx in indices_above_threshold]
    labels = [labels[idx] for idx in indices_above_threshold]

    return bboxes, scores, labels


def iob(coords1, coords2) -> float:
    return BoundingBox(*coords1).iob(BoundingBox(*coords2))


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    import torch

    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    import torch

    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append({"label": class_label, "score": float(score), "bbox": [float(elem) for elem in bbox]})

    return objects


DEFAULT_STRUCTURE_CLASS_THRESHOLDS = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10,
}


def objects_to_table(objects, tokens, structure_class_thresholds=DEFAULT_STRUCTURE_CLASS_THRESHOLDS) -> Optional[Table]:
    structures = objects_to_structures(objects, tokens=tokens, class_thresholds=structure_class_thresholds)

    if len(structures) == 0:
        return None

    cells, _ = structure_to_cells(structures, tokens=tokens)

    table_cells = []
    for cell in cells:

        rows = sorted(cell["row_nums"])
        rows = list(range(rows[0], rows[-1] + 1))

        cols = sorted(cell["column_nums"])
        cols = list(range(cols[0], cols[-1] + 1))

        table_cells.append(
            TableCell(
                content=cell["cell text"],
                rows=rows,
                cols=cols,
                is_header=cell["column header"],
                bbox=BoundingBox(*cell["bbox"]),
            )
        )

    if len(table_cells) == 0:
        return None

    return Table(table_cells)


def refine_rows(rows, tokens, score_threshold):
    """
    Apply operations to the detected rows, such as
    thresholding, NMS, and alignment.
    """

    if len(tokens) > 0:
        rows = nms_by_containment(rows, tokens, overlap_threshold=0.5)
        remove_objects_without_content(tokens, rows)
    else:
        rows = nms(rows, match_criteria="object2_overlap", match_threshold=0.5, keep_higher=True)
    if len(rows) > 1:
        rows = sort_objects_top_to_bottom(rows)

    return rows


def refine_columns(columns, tokens, score_threshold):
    """
    Apply operations to the detected columns, such as
    thresholding, NMS, and alignment.
    """

    if len(tokens) > 0:
        columns = nms_by_containment(columns, tokens, overlap_threshold=0.5)
        remove_objects_without_content(tokens, columns)
    else:
        columns = nms(columns, match_criteria="object2_overlap", match_threshold=0.25, keep_higher=True)
    if len(columns) > 1:
        columns = sort_objects_left_to_right(columns)

    return columns


def nms_by_containment(container_objects, package_objects, overlap_threshold=0.5):
    """
    Non-maxima suppression (NMS) of objects based on shared containment of other objects.
    """
    container_objects = sort_objects_by_score(container_objects)
    num_objects = len(container_objects)
    suppression = [False for obj in container_objects]

    packages_by_container, _, _ = slot_into_containers(
        container_objects,
        package_objects,
        overlap_threshold=overlap_threshold,
        unique_assignment=True,
        forced_assignment=False,
    )

    for object2_num in range(1, num_objects):
        object2_packages = set(packages_by_container[object2_num])
        if len(object2_packages) == 0:
            suppression[object2_num] = True
        for object1_num in range(object2_num):
            if not suppression[object1_num]:
                object1_packages = set(packages_by_container[object1_num])
                if len(object2_packages.intersection(object1_packages)) > 0:
                    suppression[object2_num] = True

    final_objects = [obj for idx, obj in enumerate(container_objects) if not suppression[idx]]
    return final_objects


def slot_into_containers(
    container_objects, package_objects, overlap_threshold=0.5, unique_assignment=True, forced_assignment=False
):
    """
    Slot a collection of objects into the container they occupy most
    (the container which holds the largest fraction of the object).
    """
    best_match_scores = []

    container_assignments = [[] for container in container_objects]
    package_assignments = [[] for package in package_objects]

    if len(container_objects) == 0 or len(package_objects) == 0:
        return container_assignments, package_assignments, best_match_scores

    match_scores = defaultdict(dict)
    for package_num, package in enumerate(package_objects):
        match_scores = []
        package_rect = BoundingBox(*package["bbox"])
        package_area = package_rect.area
        for container_num, container in enumerate(container_objects):
            container_rect = BoundingBox(*container["bbox"])
            intersect_area = container_rect.intersect(package_rect).area
            overlap_fraction = intersect_area / package_area
            match_scores.append({"container": container, "container_num": container_num, "score": overlap_fraction})

        sorted_match_scores = sort_objects_by_score(match_scores)

        best_match_score = sorted_match_scores[0]

        best_match_scores.append(best_match_score["score"])
        if forced_assignment or best_match_score["score"] >= overlap_threshold:
            container_assignments[best_match_score["container_num"]].append(package_num)
            package_assignments[package_num].append(best_match_score["container_num"])

        if not unique_assignment:  # slot package into all eligible slots
            for match_score in sorted_match_scores[1:]:
                if match_score["score"] >= overlap_threshold:
                    container_assignments[match_score["container_num"]].append(package_num)
                    package_assignments[package_num].append(match_score["container_num"])
                else:
                    break

    return container_assignments, package_assignments, best_match_scores


def sort_objects_by_score(objects, reverse=True):
    """
    Put any set of objects in order from high score to low score.
    """
    if reverse:
        sign = -1
    else:
        sign = 1
    return sorted(objects, key=lambda k: sign * k["score"])


def remove_objects_without_content(page_spans, objects):
    """
    Remove any objects (these can be rows, columns, supercells, etc.) that don't
    have any text associated with them.
    """
    for obj in objects[:]:
        object_text, _ = extract_text_inside_bbox(page_spans, obj["bbox"])
        if len(object_text.strip()) == 0:
            objects.remove(obj)


def extract_text_inside_bbox(spans, bbox):
    """
    Extract the text inside a bounding box.
    """
    bbox_spans = get_bbox_span_subset(spans, bbox)
    bbox_text = extract_text_from_spans(bbox_spans, remove_integer_superscripts=True)

    return bbox_text, bbox_spans


def get_bbox_span_subset(spans, bbox, threshold=0.5):
    """
    Reduce the set of spans to those that fall within a bounding box.

    threshold: the fraction of the span that must overlap with the bbox.
    """
    span_subset = []
    for span in spans:
        if overlaps(span["bbox"], bbox, threshold):
            span_subset.append(span)
    return span_subset


def overlaps(bbox1, bbox2, threshold=0.5):
    """
    Test if more than "threshold" fraction of bbox1 overlaps with bbox2.
    """
    rect1 = BoundingBox(*bbox1)
    area1 = rect1.area
    if area1 == 0:
        return False
    return rect1.intersect(BoundingBox(*bbox2)).area / area1 >= threshold


def extract_text_from_spans(spans, join_with_space=True, remove_integer_superscripts=True):
    """
    Convert a collection of page tokens/words/spans into a single text string.
    """

    if join_with_space:
        join_char = " "
    else:
        join_char = ""
    spans_copy = spans[:]

    if remove_integer_superscripts:
        for span in spans:
            if "flags" not in span:
                continue
            flags = span["flags"]
            if flags & 2**0:  # superscript flag
                if span["text"].is_integer():
                    spans_copy.remove(span)
                else:
                    span["superscript"] = True

    if len(spans_copy) == 0:
        return ""

    spans_copy.sort(key=lambda span: span["span_num"])
    spans_copy.sort(key=lambda span: span["line_num"])
    spans_copy.sort(key=lambda span: span["block_num"])

    # Force the span at the end of every line within a block to have exactly one space
    # unless the line ends with a space or ends with a non-space followed by a hyphen
    line_texts = []
    line_span_texts = [spans_copy[0]["text"]]
    for span1, span2 in zip(spans_copy[:-1], spans_copy[1:]):
        if not span1["block_num"] == span2["block_num"] or not span1["line_num"] == span2["line_num"]:
            line_text = join_char.join(line_span_texts).strip()
            if (
                len(line_text) > 0
                and not line_text[-1] == " "
                and not (len(line_text) > 1 and line_text[-1] == "-" and not line_text[-2] == " ")
            ):
                if not join_with_space:
                    line_text += " "
            line_texts.append(line_text)
            line_span_texts = [span2["text"]]
        else:
            line_span_texts.append(span2["text"])
    line_text = join_char.join(line_span_texts)
    line_texts.append(line_text)

    return join_char.join(line_texts).strip()


def sort_objects_left_to_right(objs):
    """
    Put the objects in order from left to right.
    """
    return sorted(objs, key=lambda k: k["bbox"][0] + k["bbox"][2])


def sort_objects_top_to_bottom(objs):
    """
    Put the objects in order from top to bottom.
    """
    return sorted(objs, key=lambda k: k["bbox"][1] + k["bbox"][3])


def align_columns(columns, bbox):
    """
    For every column, align the top and bottom boundaries to the final
    table bounding box.
    """
    try:
        for column in columns:
            column["bbox"][1] = bbox[1]
            column["bbox"][3] = bbox[3]
    except Exception as err:
        print("Could not align columns: {}".format(err))
        pass

    return columns


def align_rows(rows, bbox):
    """
    For every row, align the left and right boundaries to the final
    table bounding box.
    """
    try:
        for row in rows:
            row["bbox"][0] = bbox[0]
            row["bbox"][2] = bbox[2]
    except Exception as err:
        print("Could not align rows: {}".format(err))
        pass

    return rows


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure["columns"]

    # Process the headers
    column_headers = table_structure["column headers"]
    column_headers = apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process spanning cells
    spanning_cells = [elem for elem in table_structure["spanning cells"] if not elem["projected row header"]]
    projected_row_headers = [elem for elem in table_structure["spanning cells"] if elem["projected row header"]]
    spanning_cells = apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = apply_threshold(projected_row_headers, class_thresholds["table projected row header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = align_supercells(spanning_cells, rows, columns)
    spanning_cells = nms_supercells(spanning_cells)

    header_supercell_tree(spanning_cells)

    table_structure["columns"] = columns
    table_structure["rows"] = rows
    table_structure["spanning cells"] = spanning_cells
    table_structure["column headers"] = column_headers

    return table_structure


def refine_table_structures(table_bbox, table_structures, page_spans, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structures["rows"]
    columns = table_structures["columns"]

    # columns = fill_column_gaps(columns, table_bbox)
    # rows = fill_row_gaps(rows, table_bbox)

    # Process the headers
    headers = table_structures["headers"]
    headers = apply_threshold(headers, class_thresholds["table column header"])
    headers = nms(headers)
    headers = align_headers(headers, rows)

    # Process supercells
    supercells = [elem for elem in table_structures["supercells"] if not elem["subheader"]]
    subheaders = [elem for elem in table_structures["supercells"] if elem["subheader"]]
    supercells = apply_threshold(supercells, class_thresholds["table spanning cell"])
    subheaders = apply_threshold(subheaders, class_thresholds["table projected row header"])
    supercells += subheaders
    # Align before NMS for supercells because alignment brings them into agreement
    # with rows and columns first; if supercells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    supercells = align_supercells(supercells, rows, columns)
    supercells = nms_supercells(supercells)

    header_supercell_tree(supercells)

    table_structures["columns"] = columns
    table_structures["rows"] = rows
    table_structures["supercells"] = supercells
    table_structures["headers"] = headers

    return table_structures


def nms(objects, match_criteria="object2_overlap", match_threshold=0.05, keep_higher=True):
    """
    A customizable version of non-maxima suppression (NMS).

    Default behavior: If a lower-confidence object overlaps more than 5% of its area
    with a higher-confidence object, remove the lower-confidence object.

    objects: set of dicts; each object dict must have a 'bbox' and a 'score' field
    match_criteria: how to measure how much two objects "overlap"
    match_threshold: the cutoff for determining that overlap requires suppression of one object
    keep_higher: if True, keep the object with the higher metric; otherwise, keep the lower
    """
    if len(objects) == 0:
        return []

    objects = sort_objects_by_score(objects, reverse=keep_higher)

    num_objects = len(objects)
    suppression = [False for obj in objects]

    for object2_num in range(1, num_objects):
        object2_rect = BoundingBox(*objects[object2_num]["bbox"])
        object2_area = object2_rect.area
        for object1_num in range(object2_num):
            if not suppression[object1_num]:
                object1_rect = BoundingBox(*objects[object1_num]["bbox"])
                object1_area = object1_rect.area
                intersect_area = object1_rect.intersect(object2_rect).area
                try:
                    metric = 0
                    if match_criteria == "object1_overlap":
                        metric = intersect_area / object1_area
                    elif match_criteria == "object2_overlap":
                        metric = intersect_area / object2_area
                    elif match_criteria == "iou":
                        metric = intersect_area / (object1_area + object2_area - intersect_area)
                    if metric >= match_threshold:
                        suppression[object2_num] = True
                        break
                except Exception:
                    # Intended to recover from divide-by-zero
                    pass

    return [obj for idx, obj in enumerate(objects) if not suppression[idx]]


def align_headers(headers, rows) -> list[dict[str, list]]:
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers: list[dict[str, list]] = []

    for row in rows:
        row["header"] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            min_row_overlap = max(row["bbox"][1], header["bbox"][1])
            max_row_overlap = min(row["bbox"][3], header["bbox"][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect: Optional[BoundingBox] = None
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row["header"] = True
            if header_rect is None:
                header_rect = BoundingBox(*row["bbox"])
            else:
                header_rect.union_self(BoundingBox(*row["bbox"]))

            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    assert header_rect is not None
    header = {"bbox": header_rect.to_list()}
    aligned_headers.append(header)

    return aligned_headers


def align_supercells(supercells, rows, columns):
    """
    For each supercell, align it to the rows it intersects 50% of the height of,
    and the columns it intersects 50% of the width of.
    Eliminate supercells for which there are no rows and columns it intersects 50% with.
    """
    aligned_supercells = []

    for supercell in supercells:
        supercell["header"] = False
        row_bbox_rect = None
        col_bbox_rect = None
        intersecting_header_rows = set()
        intersecting_data_rows = set()
        for row_num, row in enumerate(rows):
            row_height = row["bbox"][3] - row["bbox"][1]
            supercell_height = supercell["bbox"][3] - supercell["bbox"][1]
            min_row_overlap = max(row["bbox"][1], supercell["bbox"][1])
            max_row_overlap = min(row["bbox"][3], supercell["bbox"][3])
            overlap_height = max_row_overlap - min_row_overlap
            if "span" in supercell:
                overlap_fraction = max(overlap_height / row_height, overlap_height / supercell_height)
            else:
                overlap_fraction = overlap_height / row_height
            if overlap_fraction >= 0.5:
                if "header" in row and row["header"]:
                    intersecting_header_rows.add(row_num)
                else:
                    intersecting_data_rows.add(row_num)

        # Supercell cannot span across the header boundary; eliminate whichever
        # group of rows is the smallest
        supercell["header"] = False
        if len(intersecting_data_rows) > 0 and len(intersecting_header_rows) > 0:
            if len(intersecting_data_rows) > len(intersecting_header_rows):
                intersecting_header_rows = set()
            else:
                intersecting_data_rows = set()
        if len(intersecting_header_rows) > 0:
            supercell["header"] = True
        elif "span" in supercell:
            continue  # Require span supercell to be in the header
        intersecting_rows = intersecting_data_rows.union(intersecting_header_rows)
        # Determine vertical span of aligned supercell
        for row_num in intersecting_rows:
            if row_bbox_rect is None:
                row_bbox_rect = BoundingBox(*rows[row_num]["bbox"])
            else:
                row_bbox_rect = row_bbox_rect.union_self(BoundingBox(*rows[row_num]["bbox"]))
        if row_bbox_rect is None:
            continue

        intersecting_cols = []
        for col_num, col in enumerate(columns):
            col_width = col["bbox"][2] - col["bbox"][0]
            supercell_width = supercell["bbox"][2] - supercell["bbox"][0]
            min_col_overlap = max(col["bbox"][0], supercell["bbox"][0])
            max_col_overlap = min(col["bbox"][2], supercell["bbox"][2])
            overlap_width = max_col_overlap - min_col_overlap
            if "span" in supercell:
                overlap_fraction = max(overlap_width / col_width, overlap_width / supercell_width)
                # Multiply by 2 effectively lowers the threshold to 0.25
                if supercell["header"]:
                    overlap_fraction = overlap_fraction * 2
            else:
                overlap_fraction = overlap_width / col_width
            if overlap_fraction >= 0.5:
                intersecting_cols.append(col_num)
                if col_bbox_rect is None:
                    col_bbox_rect = BoundingBox(*col["bbox"])
                else:
                    col_bbox_rect = col_bbox_rect.union_self(BoundingBox(*col["bbox"]))
        if col_bbox_rect is None:
            continue

        supercell_bbox = row_bbox_rect.intersect(col_bbox_rect).to_list()
        supercell["bbox"] = supercell_bbox

        # Only a true supercell if it joins across multiple rows or columns
        if (
            len(intersecting_rows) > 0
            and len(intersecting_cols) > 0
            and (len(intersecting_rows) > 1 or len(intersecting_cols) > 1)
        ):
            supercell["row_numbers"] = list(intersecting_rows)
            supercell["column_numbers"] = intersecting_cols
            aligned_supercells.append(supercell)

            # A span supercell in the header means there must be supercells above it in the header
            if "span" in supercell and supercell["header"] and len(supercell["column_numbers"]) > 1:
                for row_num in range(0, min(supercell["row_numbers"])):
                    new_supercell = {
                        "row_numbers": [row_num],
                        "column_numbers": supercell["column_numbers"],
                        "score": supercell["score"],
                        "propagated": True,
                    }
                    new_supercell_columns = [columns[idx] for idx in supercell["column_numbers"]]
                    new_supercell_rows = [rows[idx] for idx in supercell["row_numbers"]]
                    bbox = [
                        min([column["bbox"][0] for column in new_supercell_columns]),
                        min([row["bbox"][1] for row in new_supercell_rows]),
                        max([column["bbox"][2] for column in new_supercell_columns]),
                        max([row["bbox"][3] for row in new_supercell_rows]),
                    ]
                    new_supercell["bbox"] = bbox
                    aligned_supercells.append(new_supercell)

    return aligned_supercells


def nms_supercells(supercells):
    """
    A NMS scheme for supercells that first attempts to shrink supercells to
    resolve overlap.
    If two supercells overlap the same (sub)cell, shrink the lower confidence
    supercell to resolve the overlap. If shrunk supercell is empty, remove it.
    """

    supercells = sort_objects_by_score(supercells)
    num_supercells = len(supercells)
    suppression = [False for supercell in supercells]

    for supercell2_num in range(1, num_supercells):
        supercell2 = supercells[supercell2_num]
        for supercell1_num in range(supercell2_num):
            supercell1 = supercells[supercell1_num]
            remove_supercell_overlap(supercell1, supercell2)
        if (
            (len(supercell2["row_numbers"]) < 2 and len(supercell2["column_numbers"]) < 2)
            or len(supercell2["row_numbers"]) == 0
            or len(supercell2["column_numbers"]) == 0
        ):
            suppression[supercell2_num] = True

    return [obj for idx, obj in enumerate(supercells) if not suppression[idx]]


def header_supercell_tree(supercells):
    """
    Make sure no supercell in the header is below more than one supercell in any row above it.
    The cells in the header form a tree, but a supercell with more than one supercell in a row
    above it means that some cell has more than one parent, which is not allowed. Eliminate
    any supercell that would cause this to be violated.
    """
    header_supercells = [supercell for supercell in supercells if "header" in supercell and supercell["header"]]
    header_supercells = sort_objects_by_score(header_supercells)

    for header_supercell in header_supercells[:]:
        ancestors_by_row = defaultdict(int)
        min_row = min(header_supercell["row_numbers"])
        for header_supercell2 in header_supercells:
            max_row2 = max(header_supercell2["row_numbers"])
            if max_row2 < min_row:
                if set(header_supercell["column_numbers"]).issubset(set(header_supercell2["column_numbers"])):
                    for row2 in header_supercell2["row_numbers"]:
                        ancestors_by_row[row2] += 1
        for row in range(0, min_row):
            if not ancestors_by_row[row] == 1:
                supercells.remove(header_supercell)
                break


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj["label"] == "table"]

    assert len(tables) <= 1
    if len(tables) == 0:
        return {}

    table = tables[0]
    structure = {}
    # for table in tables:

    table_objects = [obj for obj in objects if iob(obj["bbox"], table["bbox"]) >= 0.5]
    table_tokens = [token for token in tokens if iob(token["bbox"], table["bbox"]) >= 0.5]
    # table_tokens = []

    columns = [obj for obj in table_objects if obj["label"] == "table column"]
    rows = [obj for obj in table_objects if obj["label"] == "table row"]
    column_headers = [obj for obj in table_objects if obj["label"] == "table column header"]
    spanning_cells = [obj for obj in table_objects if obj["label"] == "table spanning cell"]

    for obj in spanning_cells:
        obj["projected row header"] = False
    projected_row_headers = [obj for obj in table_objects if obj["label"] == "table projected row header"]
    for obj in projected_row_headers:
        obj["projected row header"] = True
    spanning_cells += projected_row_headers
    for obj in rows:
        obj["column header"] = False
        for header_obj in column_headers:
            if iob(obj["bbox"], header_obj["bbox"]) >= 0.5:
                obj["column header"] = True

    # Refine table structures
    rows = refine_rows(rows, table_tokens, class_thresholds["table row"])
    columns = refine_columns(columns, table_tokens, class_thresholds["table column"])

    # Shrink table bbox to just the total height of the rows
    # and the total width of the columns
    if len(rows) > 0:
        row_rect = BoundingBox(*rows[0]["bbox"])
        for obj in rows:
            row_rect.union_self(BoundingBox(*obj["bbox"]))
    else:
        row_rect = BoundingBox(0, 0, 0, 0)

    if len(columns) > 0:
        column_rect = BoundingBox(*columns[0]["bbox"])
        for obj in columns:
            column_rect.union_self(BoundingBox(*obj["bbox"]))
    else:
        column_rect = BoundingBox(0, 0, 0, 0)

    table["row_column_bbox"] = [column_rect.x1, row_rect.y1, column_rect.x2, row_rect.y2]
    table["bbox"] = table["row_column_bbox"]

    # Process the rows and columns into a complete segmented table
    columns = align_columns(columns, table["row_column_bbox"])
    rows = align_rows(rows, table["row_column_bbox"])

    structure["rows"] = rows
    structure["columns"] = columns
    structure["column headers"] = column_headers
    structure["spanning cells"] = spanning_cells

    if len(rows) > 0 and len(columns) > 1:
        structure = refine_table_structure(structure, class_thresholds)

    return structure


def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    columns = table_structure["columns"]
    rows = table_structure["rows"]
    spanning_cells = table_structure["spanning cells"]
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = BoundingBox(*column["bbox"])
            row_rect = BoundingBox(*row["bbox"])
            cell_rect = row_rect.intersect(column_rect)
            header = "column header" in row and row["column header"]
            cell = {
                "bbox": cell_rect.to_list(),
                "column_nums": [column_num],
                "row_nums": [row_num],
                "column header": header,
            }

            cell["subcell"] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = BoundingBox(*spanning_cell["bbox"])
                if (spanning_cell_rect.intersect(cell_rect).area / cell_rect.area) > 0.5:
                    cell["subcell"] = True
                    break

            if cell["subcell"]:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell["projected row header"] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = BoundingBox(*spanning_cell["bbox"])
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        header = True
        for subcell in subcells:
            subcell_rect = BoundingBox(*subcell["bbox"])
            subcell_rect_area = subcell_rect.area
            if (subcell_rect.intersect(spanning_cell_rect).area / subcell_rect_area) > 0.5:
                if cell_rect is None:
                    cell_rect = BoundingBox(*subcell["bbox"])
                else:
                    cell_rect.union_self(BoundingBox(*subcell["bbox"]))
                cell_rows = cell_rows.union(set(subcell["row_nums"]))
                cell_columns = cell_columns.union(set(subcell["column_nums"]))
                # By convention here, all subcells must be classified
                # as header cells for a spanning cell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                header = header and "column header" in subcell and subcell["column header"]
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {
                "bbox": cell_rect.to_list(),
                "column_nums": list(cell_columns),
                "row_nums": list(cell_rows),
                "column header": header,
                "projected row header": spanning_cell["projected row header"],
            }
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except Exception:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = BoundingBox.from_union(
            BoundingBox(*dilated_columns[column_num]["bbox"]) for column_num in cell["column_nums"]
        )

        row_rect = BoundingBox.from_union(BoundingBox(*dilated_rows[row_num]["bbox"]) for row_num in cell["row_nums"])
        cell_rect = column_rect.intersect(row_rect)
        cell["bbox"] = cell_rect.to_list()

    span_nums_by_cell, _, _ = slot_into_containers(
        cells, tokens, overlap_threshold=0.001, unique_assignment=True, forced_assignment=False
    )

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]

        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell["cell text"] = extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell["spans"] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = sort_objects_top_to_bottom(rows)

    num_columns = len(columns)
    columns = sort_objects_left_to_right(columns)

    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell["spans"]:
            min_x_values_by_column[min_column].append(span["bbox"][0])
            min_y_values_by_row[min_row].append(span["bbox"][1])
            max_x_values_by_column[max_column].append(span["bbox"][2])
            max_y_values_by_row[max_row].append(span["bbox"][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row["bbox"][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row["bbox"][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row["bbox"][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row["bbox"][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column["bbox"][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column["bbox"][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column["bbox"][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column["bbox"][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = BoundingBox.from_union(BoundingBox(*rows[row_num]["bbox"]) for row_num in cell["row_nums"])
        column_rect = BoundingBox.from_union(
            BoundingBox(*columns[column_num]["bbox"]) for column_num in cell["column_nums"]
        )

        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.area > 0:
            cell["bbox"] = cell_rect.to_list()

    return cells, confidence_score


def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
        num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell["column header"]]
    if len(header_cells) > 0:
        max_header_row = max([max(cell["row_nums"]) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell["row_nums"]:
                for column_num in cell["column_nums"]:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[: max_header_row + 1, :]

    flattened_header = []

    for col in header.transpose():
        flattened_header.append(" | ".join(OrderedDict.fromkeys(col)))

    df = pd.DataFrame(table_array[max_header_row + 1 :, :], index=None, columns=flattened_header)

    return df.to_csv(index=None)


def cells_to_html(cells):
    cells = sorted(cells, key=lambda k: min(k["column_nums"]))
    cells = sorted(cells, key=lambda k: min(k["row_nums"]))

    table = ET.Element("table")
    current_row = -1

    for cell in cells:
        this_row = min(cell["row_nums"])

        attrib = {}
        colspan = len(cell["column_nums"])
        if colspan > 1:
            attrib["colspan"] = str(colspan)
        rowspan = len(cell["row_nums"])
        if rowspan > 1:
            attrib["rowspan"] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if cell["column header"]:
                cell_tag = "th"
                row = ET.SubElement(table, "thead")
            else:
                cell_tag = "td"
                row = ET.SubElement(table, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell["cell text"]

    return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))


def remove_supercell_overlap(supercell1, supercell2):
    """
    This function resolves overlap between supercells (supercells must be
    disjoint) by iteratively shrinking supercells by the fewest grid cells
    necessary to resolve the overlap.
    Example:
    If two supercells overlap at grid cell (R, C), and supercell #1 is less
    confident than supercell #2, we eliminate either row R from supercell #1
    or column C from supercell #1 by comparing the number of columns in row R
    versus the number of rows in column C. If the number of columns in row R
    is less than the number of rows in column C, we eliminate row R from
    supercell #1. This resolves the overlap by removing fewer grid cells from
    supercell #1 than if we eliminated column C from it.
    """
    common_rows = set(supercell1["row_numbers"]).intersection(set(supercell2["row_numbers"]))
    common_columns = set(supercell1["column_numbers"]).intersection(set(supercell2["column_numbers"]))

    # While the supercells have overlapping grid cells, continue shrinking the less-confident
    # supercell one row or one column at a time
    while len(common_rows) > 0 and len(common_columns) > 0:
        # Try to shrink the supercell as little as possible to remove the overlap;
        # if the supercell has fewer rows than columns, remove an overlapping column,
        # because this removes fewer grid cells from the supercell;
        # otherwise remove an overlapping row
        if len(supercell2["row_numbers"]) < len(supercell2["column_numbers"]):
            min_column = min(supercell2["column_numbers"])
            max_column = max(supercell2["column_numbers"])
            if max_column in common_columns:
                common_columns.remove(max_column)
                supercell2["column_numbers"].remove(max_column)
            elif min_column in common_columns:
                common_columns.remove(min_column)
                supercell2["column_numbers"].remove(min_column)
            else:
                supercell2["column_numbers"] = []
                common_columns = set()
        else:
            min_row = min(supercell2["row_numbers"])
            max_row = max(supercell2["row_numbers"])
            if max_row in common_rows:
                common_rows.remove(max_row)
                supercell2["row_numbers"].remove(max_row)
            elif min_row in common_rows:
                common_rows.remove(min_row)
                supercell2["row_numbers"].remove(min_row)
            else:
                supercell2["row_numbers"] = []
                common_rows = set()
