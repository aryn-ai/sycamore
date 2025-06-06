"""
# Reliable opensearch re-loading

Goal: A fast way of answering the question "which of these documents in a set of materialize
directories need to be added or removed from this opensearch index?"

Assumptions:
* List (w/o attributes) is relatively fast on all platforms.
* Total list of items can be stores in RAM on one node
* The source directories are the end of a reliable materialize pipeline so each document may need
  to be split into sub-pieces.
* The splitting operation doesn't care about the document id of the split pieces
* Reading an object (even a small one) is much slower that listing
* Systems allow for moderately long names (multiple encoded sha256's)
* DocIDs for the source docs are unique and not re-used (so in particular updating the document
  should change the docid)
* Document writes into opensearch are atomic.

So the trick is that we're going to encode everything we need to know to determine if we need to
reload in the names in both the source and the destination. So the algorithm for determining what
to update is:

1. List the source directory
2. List the destination opensearch index fetching the parent_id and doc_id
3. Split the source directory into:
   a. base files: path-sha256-<doc_id>.pickle
      -- TODO: fix the doc_ids in these to be more compactly encoded
   b. Find all of the md-rl-<doc_id>-<key>.pickle files in the source directory
4. For every file that is missing it's associated md-rl-<doc_id> file, add that document to the
   to_be_loaded list.
5. For every file that as a md-rl-<doc_id> file, find all the destination rows where doc_id is
   parent_id or doc_id, sort that list, and calculate the sorted hash. Add the document to the
   to_be_loaded list if it doesn't match.
6. For every indexed-doc where the doc_id or parent_id is not in the source, add the document to
   the to_be_deleted list; similarly for md-rl documents missing their associated base document.

Loading a document
1. Given a document and the associated indexed-docs (if any):
2. Run the divide/split/explode function associated with the document to get the documents that
   will be loaded. Transform them into their too-be-loaded form (without the id) and calculate
   a content hash over the data; include the position in the split so that even duplicate content
   has a different hash.
3. Assign te content hashes except to the potentially 1 "original" document which should be the
   first value returned in the list.  That document keeps the original doc_id, the others get
   doc_ids from their content hashes.
4. Sort the index-document ids, and calculate the hash over those to get the <key>.
5. Write the associated md-rl object.
6. Delete any index documents that were present but not in the list; warn on this state since
   it implies tat the split function is non-deterministic.
7. Create any new objects that were missing; skip any that were already present we know from
   the content-hash that they don't need to be updated.

Checking for a reliable load
1. Reprocess like at the start, but then just verify that every document in the to_be_added list
   is recent, and every document in the to_be_deleted list has a md-rl document.

Notes:
1. For efficiency it's best to be running at most one of these processes/opensearch db; it will be
   correct but inefficient if that isn't true.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import logging
import random
import re
import time
from typing import Callable, Tuple, TYPE_CHECKING

from sycamore.data.document import Document
from sycamore.connectors.opensearch.opensearch_writer import (
    OpenSearchWriterClientParams,
    OpenSearchWriterTargetParams,
    get_existing_target_params,
    OpenSearchWriterRecord,
)
from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# TODO: figure out how best to unify with materialize.py:_PyArrowFsHelper
class _MatDir:
    def __init__(self, path):
        from sycamore.utils.pyarrow import infer_fs

        self.fs, self.root = infer_fs(path)

    def list_files(self):
        from pyarrow.fs import FileSelector

        return self.fs.get_file_info(FileSelector(self.root, recursive=False))

    def read_document(self, filename):
        assert "/" not in filename, f"{filename} should be just the name"
        path = f"{self.root}/{filename}"
        with self.fs.open_input_stream(str(path)) as f:
            return Document.deserialize(f.read())

    def touch_file(self, filename):
        assert "/" not in filename, f"{filename} should be just the name"
        path = f"{self.root}/{filename}"
        with self.fs.open_output_stream(path, compression=None):
            pass


def raw_id_to_doc_id(raw_id):
    return f"path-sha256-{raw_id}"


def raw_id_to_filename(raw_id):
    return f"doc-{raw_id_to_doc_id(raw_id)}.pickle"


def calculate_doc_key(parts):
    parts.sort()
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return base64.urlsafe_b64encode(h.digest()).decode("UTF-8")


KEYSEP = ","  # must not be be in ID_RE
ID_RE = "[-0-9a-zA-Z_=]+"  # URLsafe base64, also accepts old hex format


# Todo accept sources as docset and require it end with materialize.
class OpenSearchSync:
    def __init__(
        self,
        sources: list[Tuple[str, Callable[[Document], list[Document]]]],
        client_params: OpenSearchWriterClientParams,
        target_params: OpenSearchWriterTargetParams,
    ):
        self.sources = sources
        assert isinstance(client_params, OpenSearchWriterClientParams)
        self.os_client_params = asdict(client_params)
        assert isinstance(target_params, OpenSearchWriterTargetParams)
        self.target_params = target_params

    def sync(self):
        with ThreadPoolExecutor() as e:
            os_files_fut = e.submit(self.prepare_opensearch)
            # sychronous
            source_files = list(e.map(self.find_source_files, self.sources))

            # mtime is present it's just not printed, and the interface is a mess since
            # according to docs, mtime can be a datetime or a float, or mtime_ns can be present
            # docs lie for a local filesystem, both are present
            os_pid_to_key = os_files_fut.result()

            # TODO: once we have delete implemented:
            # If we add the timestamp from the filesystem into the calculated key; and we implement
            # deletion on key mismatch, then we can handle the case where files get updated in
            # ETL, or we change which mat dir is synced to an index.
            # If delete also removes the oss-<did>,key.md file; then we can even handle the
            # case of update in place.

            # need to track source to know the splitter
            to_be_loaded_groups = [[] for s in self.sources]
            for i, x in enumerate(source_files):
                fs, id_to_key = x
                for f in fs:
                    if f not in id_to_key:  # no filesystem md record, must reload
                        to_be_loaded_groups[i].append(f)
                    elif f not in os_pid_to_key:  # no os record, must reload
                        to_be_loaded_groups[i].append(f)
                    elif id_to_key[f] == os_pid_to_key[f]:
                        pass  # in opensearch and calculated key matches expected
                    else:
                        # Might want to delete in this case. If there is a non-determinism
                        # in calculating stuff (which shouldn't happen), then we could get in
                        # the state where we never successfully load the document since we would
                        # have partial, incorrect stuff in there
                        logger.info(
                            f"Mismatch on document id {f}: filesystem_key={id_to_key[f]} os_calculated_key={os_pid_to_key[f]}"
                        )
                        to_be_loaded_groups[i].append(f)

            for i, g in enumerate(to_be_loaded_groups):
                root, splitter = self.sources[i]
                self.load_batch(root, splitter, g)

    def find_source_files(self, src):
        base_re = re.compile(f"doc-path-sha256-({ID_RE})\\.pickle")
        # does not match with *.pickle so materialize will ignore
        oss_md_re = re.compile(f"^oss-({ID_RE}){KEYSEP}({ID_RE})\\.md")
        path, splitter = src
        fis = _MatDir(path).list_files()

        base_files = []
        id_to_key = {}

        for f in fis:
            assert f.is_file, f"{f.base_name} is not a file, but mat dir should be all files"
            if f.base_name.startswith("materialize."):
                continue
            elif m := base_re.fullmatch(f.base_name):
                base_files.append(m.group(1))
                assert raw_id_to_filename(m.group(1)) == f.base_name
            elif m := oss_md_re.fullmatch(f.base_name):
                id_to_key[m.group(1)] = m.group(2)
            else:
                assert False, f"Should not have an unexpected file like {f.base_name}"

        return [base_files, id_to_key]

    def prepare_opensearch(self):
        """Make sure the index exists in a compatible way and
        find files in the index."""
        from opensearchpy.exceptions import NotFoundError

        def process_hits(os_docs, response):
            for h in response["hits"]["hits"]:
                os_docs.append(
                    {
                        "doc_id": h["_id"],
                        "parent_id": h["_source"].get("parent_id", None),
                    }
                )

        with self.os_client() as os:
            tp = self.target_params
            try:
                response = os.search(
                    index=tp.index_name,
                    body={
                        "query": {"match_all": {}},
                        "_source": ["parent_id"],
                    },
                    scroll="1m",
                    size=1000,
                )
            except NotFoundError:
                logger.info(f"index {tp.index_name} not found. Creating it")
                os.indices.create(tp.index_name, body={"mappings": tp.mappings, "settings": tp.settings})
                return {}

            existing_params = get_existing_target_params(os, tp)
            assert tp.compatible_with(existing_params)
            scroll_id = response["_scroll_id"]
            os_docs = []
            process_hits(os_docs, response)
            if scroll_id is not None:
                while len(response["hits"]["hits"]) > 0:
                    logger.info(f"Processed batch, {len(os_docs)} so far...")
                    response = os.scroll(scroll_id=scroll_id, scroll="1m")
                    process_hits(os_docs, response)

                os.clear_scroll(scroll_id=scroll_id)

        pid_to_parts = {}
        for o in os_docs:
            assert "doc_id" in o
            if o["parent_id"] is None:
                did = o["doc_id"]
                assert did.startswith("path-sha256-"), f"opensearch document {did} with no parent has incorrect prefix"
                pid = did.removeprefix("path-sha256-")
                pid_to_parts.setdefault(pid, []).append("root")
            else:
                did, pid = o["doc_id"], o["parent_id"]
                assert did.startswith("splitdoc-"), f"opensearch document {did} with parent {pid} has incorrect prefix"
                assert pid.startswith(
                    "path-sha256-"
                ), f"opensearch document {did} with parent {pid} has incorrect parent prefix"
                did = did.removeprefix("splitdoc-")
                pid = pid.removeprefix("path-sha256-")

                pid_to_parts.setdefault(pid, []).append(did)

        for k in pid_to_parts:
            pid_to_parts[k] = calculate_doc_key(pid_to_parts[k])

        return pid_to_parts

    def os_client(self):
        assert False
        return OpenSearchClientWithLogging(**self.os_client_params)

    def load_batch(self, root, splitter, ids):
        with self.os_client() as os:
            self.LoadBatch(root, splitter, ids, os, self.target_params).run()

    class LoadBatch:
        def __init__(self, root, splitter, ids, os_client, target_params):
            self.root = root
            self.splitter = splitter
            self.ids = ids
            self.os_client = os_client
            self.target_params = target_params

            self.retry_count = 0
            self.os_batch_size = 100
            self.records = []
            self.pending_successful_write = {}
            self.id_to_parent_id = {}

        def run(self):
            mat_dir = _MatDir(self.root)

            for i in self.ids:
                fn = raw_id_to_filename(i)
                doc = mat_dir.read_document(fn)

                self.records.extend(self.split_doc(doc, i))
                if len(self.records) >= self.os_batch_size:
                    self.write_os_records()

            while len(self.records) > 0:
                self.write_os_records(mat_dir)

        def split_doc(self, doc, expected_raw_id) -> list[OpenSearchWriterRecord]:
            expected_doc_id = raw_id_to_doc_id(expected_raw_id)
            short_doc_id = expected_doc_id.removeprefix("path-sha256-")
            psw: dict[str, bool] = {}
            self.pending_successful_write[short_doc_id] = psw

            parent_id = doc.doc_id
            assert "/" not in parent_id, f"split character '/' can not be in doc id {parent_id}"
            assert (
                parent_id == expected_doc_id
            ), f"mismatch between document id {parent_id} and filename id {expected_doc_id}"
            parts = self.splitter(doc)
            assert isinstance(parts, list), "splitter did not return list"
            assert len(parts) > 0, "splitter returned empty list"
            for p in parts:
                assert isinstance(p, Document), "splitter returned non-document"
            ret = []
            if parts[0].parent_id is None:
                assert (
                    parts[0].doc_id == parent_id
                ), f"If first doc has no parent id, it should still have doc_id as its id, but {parts[0].doc_id} != {parent_id}"
                psw["root"] = True
                ret.append(OpenSearchWriterRecord.from_doc(parts[0], self.target_params))
                parts = parts[1:]

            for i, p in enumerate(parts):
                assert (
                    p.parent_id == parent_id
                ), f"Subdocs should have proper parent id. got {p.parent_id}; want {parent_id}"
                p.doc_id = "X"
                r = OpenSearchWriterRecord.from_doc(p, self.target_params)
                assert r._id == "X" and r._source["doc_id"] == "X"
                h = hashlib.sha256(f"{parent_id}/{i}/".encode("UTF-8"))
                # sort keys so the hash should be deterministic.
                h.update(json.dumps(r._source, sort_keys=True).encode("UTF-8"))
                sid = base64.urlsafe_b64encode(h.digest()).decode("UTF-8")
                r._id = "splitdoc-" + sid
                r._source["doc_id"] = r._id
                ret.append(r)
                psw[sid] = True
                self.id_to_parent_id[sid] = short_doc_id

            psw["key"] = calculate_doc_key(list(psw.keys()))
            return ret

        def write_os_records(self, mat_dir):
            def generate_records(records):
                for r in records:
                    yield asdict(r)

            retry_records = []
            for success, item in self.os_client.parallel_bulk(
                generate_records(self.records), **self.target_params.insert_settings
            ):
                if success:
                    did = item["index"]["_id"]
                    if did.startswith("path-sha256-"):
                        did = did.removeprefix("path-sha256-")
                        assert (
                            did in self.pending_successful_write
                        ), f"root doc {did} has no entry in pending_successful_write?"
                        psw = self.pending_successful_write[did]
                        assert (
                            "root" in psw
                        ), f"root of docid {did} successfully written, but not a pending successful write?"
                        del psw["root"]
                    elif did.startswith("splitdoc-"):
                        cdid = did.removeprefix("splitdoc-")
                        assert (
                            cdid in self.id_to_parent_id
                        ), f"doc id {cdid} successfully written, but not in id_to_parent_id"
                        did = self.id_to_parent_id.pop(cdid)
                        assert (
                            did in self.pending_successful_write
                        ), f"parent doc {did} has no entry in pending_successful_write?"
                        psw = self.pending_successful_write[did]
                        assert cdid in psw, f"sub-doc {cdid} of {did} not present in pending_successful_write"
                        del psw[cdid]
                    else:
                        raise Exception(f"unexpected doc_id {did} in successful write")

                    assert len(psw) >= 1, f"psw for {did} has len 0"
                    if len(psw) == 1:
                        assert "key" in psw
                        path = f"oss-{did}{KEYSEP}{psw['key']}.md"
                        logger.debug(f"Successfully wrote all parts of {did} touching {path}")
                        mat_dir.touch_file(path)

                    pass
                elif item["index"]["status"] == 429:
                    retry_records.append(item["index"]["data"])
                else:
                    msg = f"Failed to upload documnet: {item}"
                    logger.error(msg)
                    raise Exception(msg)

            if len(retry_records) == 0:
                self.retry_count = 0
            else:
                self.backoff()

            self.records = retry_records

        def backoff(self):
            if self.retry_count > 6:
                raise Exception("too many consecutive requests that required backoff")

            backoff = 1 * (2**self.retry_count)
            sleep_time = backoff + random.uniform(0, 0.1 * backoff)
            self.retry_count += 1

            logger.warning(f"{self.retry_count} consecutive requests with some 429s. Sleep({sleep_time:%.2f}).")
            time.sleep(sleep_time)
