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

# TODO: rethink the constant translation between short ids (hash only) and the full doc id
# (path-sha256-<hash> or splitdoc-<hash>). With deletion it's not really saving memory since we
# need the full doc id to delete in opensearch (or another way to tell what type of docid it is),
# and it leads to a lot of back-and-forth translation in the code. Conventiently if we switch it,
# the hash changes will force it to auto-clean-up.

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import logging
import random
import re
import time
from typing import Any, Callable, Tuple, TYPE_CHECKING

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


def doc_to_record(d: Document, target_params: OpenSearchWriterTargetParams) -> OpenSearchWriterRecord:
    r = OpenSearchWriterRecord.from_doc(d, target_params)
    if "doc_mtime" in d.data:
        r._source["doc_mtime"] = d.data["doc_mtime"]

    return r


# TODO: figure out how best to unify with materialize.py:_PyArrowFsHelper
class _MatDir:
    def __init__(self, path):
        from sycamore.utils.pyarrow import infer_fs

        self.fs, self.root = infer_fs(path)

    def list_files(self):
        from pyarrow.fs import FileSelector

        return self.fs.get_file_info(FileSelector(self.root, recursive=False))

    def read_document(self, filename):
        path = self._name_to_path(filename)
        with self.fs.open_input_stream(str(path)) as f:
            return Document.deserialize(f.read())

    def touch_file(self, filename):
        path = self._name_to_path(filename)
        with self.fs.open_output_stream(path, compression=None):
            pass

    def delete_file(self, filename):
        self.fs.delete_file(self._name_to_path(filename))

    def _name_to_path(self, filename):
        assert "/" not in filename, f"{filename} should be just the name"
        return f"{self.root}/{filename}"


def raw_id_to_doc_id(raw_id):
    return f"path-sha256-{raw_id}"


def raw_id_to_filename(raw_id):
    return f"doc-{raw_id_to_doc_id(raw_id)}.pickle"


def calculate_doc_key(mtime, parts):
    parts.sort()
    h = hashlib.sha256()
    h.update(mtime.to_bytes(8, "big", signed=True))
    for p in parts:
        h.update(p.encode())
    return base64.urlsafe_b64encode(h.digest()).decode("UTF-8")


KEYSEP = ","  # must not be be in ID_RE
ID_RE = "[-0-9a-zA-Z_=]+"  # URLsafe base64, also accepts old hex format


# Todo accept sources as docset and require it end with materialize.
class OpenSearchSync:
    """Note: The Callable in sources is a way of taking a document and turning it into a bunch of
    sub-documents that should be loaded into OpenSearch. For documents read through the partitioner
    this is likely to be explode. For other types of documents, e.g. JSON, it could be a custom
    function that relies on the JSON value.
    """

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
        self.stats = SyncStats()

    def sync(self) -> None:
        self.stats = SyncStats()
        with ThreadPoolExecutor() as e:
            os_files_fut = e.submit(self.prepare_opensearch)
            # sychronous
            source_files = list(e.map(self.find_source_files, self.sources))

            # mtime is present it's just not printed, and the interface is a mess since
            # according to docs, mtime can be a datetime or a float, or mtime_ns can be present
            # docs lie for a local filesystem, both are present
            os_pid_to_parts = os_files_fut.result()

            if False:
                from devtools import PrettyFormat

                print(f"-------DEBUGGING-------\nSource_Files\n{PrettyFormat()(source_files)}")
                print(f"-------DEBUGGING-------\nOS PidToParts\n{PrettyFormat()(os_pid_to_parts)}")

            # need to track source to know the splitter
            to_be_loaded_groups: list[list[str]] = [[] for s in self.sources]
            all_source_ids: set[str] = set()
            changed_pid_to_osids = {}
            for i, sf in enumerate(source_files):
                all_source_ids.update(sf.id_to_key.keys())
                self.stats.updated_source_file += sf.updated_count
                for f in sf.fid_to_mtime:
                    if f not in sf.id_to_key:  # no filesystem md record, must reload
                        to_be_loaded_groups[i].append(f)
                        self.stats.missing_md_info += 1
                    elif f not in os_pid_to_parts:  # no os record, must reload
                        to_be_loaded_groups[i].append(f)
                        self.stats.missing_os_record += 1
                    elif sf.id_to_key[f] == os_pid_to_parts[f]["key"]:
                        # in opensearch and calculated key matches expected
                        self.stats.correctly_loaded += 1
                    else:
                        self.stats.mismatch_key += 1
                        # Deleting in this case, either we loaded something incorrectly, or
                        # the source document changed in a way that caused the key to change
                        # in either case, we need to delete what is in opensearch and replace
                        # it with updated information.
                        os_ids = os_pid_to_parts[f]["os_ids"]
                        changed_pid_to_osids[f] = os_ids
                        logger.info(
                            f"Mismatch on document id {f}: filesystem_key={sf.id_to_key[f]} os_calculated_key={os_pid_to_parts[f]['key']}; os_ids={os_ids}"
                        )
                        to_be_loaded_groups[i].append(f)

            self.delete_os_not_in_source(os_pid_to_parts, all_source_ids)

            for i, g in enumerate(to_be_loaded_groups):
                # See comment on class for what splitter is
                root, splitter = self.sources[i]
                fid_to_mtime = source_files[i].fid_to_mtime
                with self.os_client() as os:
                    self.ProcessBatch(
                        root, splitter, g, fid_to_mtime, changed_pid_to_osids, os, self.target_params
                    ).run()

    def find_source_files(self, src: Tuple[str, Callable[[Document], list[Document]]]) -> "SourceFileInfo":
        """Finds source file information, also cleans up files which can't be part of a source"""
        base_re = re.compile(f"doc-path-sha256-({ID_RE})\\.pickle")
        # does not match with *.pickle so materialize will ignore
        oss_md_re = re.compile(f"^oss-({ID_RE}){KEYSEP}(\\d+){KEYSEP}({ID_RE})\\.md")
        path, splitter = src
        mat_dir = _MatDir(path)
        fis = mat_dir.list_files()

        fid_to_mtime = {}
        id_to_info: dict[str, Any] = {}  # should be a list or a tuple, but I can't make mypy happy

        updated_count = 0

        for f in fis:
            assert f.is_file, f"{f.base_name} is not a file, but mat dir should be all files"
            if f.base_name.startswith("materialize."):
                continue
            elif m := base_re.fullmatch(f.base_name):
                mtime_ns = f.mtime_ns
                if mtime_ns is None:
                    assert f.mtime is not None
                    mtime_ns = int(1e9 * f.mtime)
                fid_to_mtime[m.group(1)] = mtime_ns
                assert raw_id_to_filename(m.group(1)) == f.base_name
            elif m := oss_md_re.fullmatch(f.base_name):
                did, mtime, key = m.group(1), int(m.group(2)), m.group(3)
                if did in id_to_info:
                    logger.warning(f"Duplicate key for {did} {id_to_info[did]} {mtime} {key}")
                    if not isinstance(id_to_info[did], list):
                        a = id_to_info[did]
                        assert isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], int) and isinstance(a[1], str)
                        id_to_info[did] = [a]
                    id_to_info[did].append((mtime, key))
                else:
                    id_to_info[did] = (mtime, key)
            elif f.base_name.startswith("oss-") and f.base_name.endswith(".md"):
                # Clean these up; likely cause is a change to the format.
                logger.warning(f"Unexpected mis-formatted oss-*.md file {f.base_name} found")
                self.stats.misformatted_oss_file += 1
                mat_dir.delete_file(f.base_name)
            elif f.base_name.endswith(".md"):
                self.stats.ignored_other_md += 1
            elif f.base_name.endswith(".pickle"):
                raise ValueError(
                    f"Found file {f.base_name} ending with .pickle; is this not a reliable pipeline? doc ids should start with doc-path-sha256-"
                )
            else:
                self.stats.ignored_unrecognized += 1
                logger.warning(f"Ignoring unrecognized, file {f.base_name}")

        to_remove = {}
        for k, v in id_to_info.items():
            if k not in fid_to_mtime or isinstance(v, list):
                to_remove[k] = v
            elif v[0] != fid_to_mtime[k]:
                updated_count += 1
                to_remove[k] = v

        if len(to_remove) > 0:
            lingering_metadata = len(to_remove) - updated_count
            logger.info(f"{lingering_metadata} lingering oss-metadata files, {updated_count} updated files")
            for k, v in to_remove.items():
                del id_to_info[k]
                if isinstance(v, list):
                    for w in v:
                        mat_dir.delete_file(f"oss-{k}{KEYSEP}{w[0]}{KEYSEP}{w[1]}.md")
                else:
                    mat_dir.delete_file(f"oss-{k}{KEYSEP}{v[0]}{KEYSEP}{v[1]}.md")

        # turn it into id_to_key; definitive mtimes, including for new files are in
        # fid_to_mtime
        for k, v in id_to_info.items():
            id_to_info[k] = v[1]

        return SourceFileInfo(fid_to_mtime, id_to_info, updated_count)

    def prepare_opensearch(self) -> dict[str, dict]:
        """Make sure the index exists in a compatible way and
        find files in the index."""
        from opensearchpy.exceptions import NotFoundError

        def process_hits(os_docs, response):
            for h in response["hits"]["hits"]:
                d = {
                    "doc_id": h["_id"],
                    "parent_id": h["_source"].get("parent_id", None),
                }
                if (mtime := h["_source"].get("doc_mtime", None)) is not None:
                    d["doc_mtime"] = mtime

                os_docs.append(d)

        with self.os_client() as os:
            tp = self.target_params
            try:
                response = os.search(
                    index=tp.index_name,
                    body={
                        "query": {"match_all": {}},
                        "_source": ["parent_id", "doc_mtime"],
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
            os_docs: list[dict] = []
            process_hits(os_docs, response)
            if scroll_id is not None:
                while len(response["hits"]["hits"]) > 0:
                    logger.info(f"Processed batch, {len(os_docs)} so far...")
                    response = os.scroll(scroll_id=scroll_id, scroll="1m")
                    process_hits(os_docs, response)

                os.clear_scroll(scroll_id=scroll_id)

        pid_to_parts: dict[str, dict] = {}
        for o in os_docs:
            assert "doc_id" in o
            if o["parent_id"] is None:
                did = o["doc_id"]
                assert did.startswith("path-sha256-"), f"opensearch document {did} with no parent has incorrect prefix"
                pid = did.removeprefix("path-sha256-")
                did = pid
            else:
                did, pid = o["doc_id"], o["parent_id"]
                assert did.startswith("splitdoc-"), f"opensearch document {did} with parent {pid} has incorrect prefix"
                assert pid.startswith(
                    "path-sha256-"
                ), f"opensearch document {did} with parent {pid} has incorrect parent prefix"
                did = did.removeprefix("splitdoc-")
                pid = pid.removeprefix("path-sha256-")

            parts = pid_to_parts.setdefault(pid, {"parts": [], "os_ids": []})
            parts["parts"].append(did)
            parts["os_ids"].append(o["doc_id"])
            if (mtime := o.get("doc_mtime", None)) is not None:
                if "doc_mtime" in parts:
                    logger.warning(
                        f"Incorrect duplicate doc_mtime in multiple os docs for {pid}; values {mtime} {parts['doc_mtime']}"
                    )
                    parts["doc_mtime"] = -1
                else:
                    parts["doc_mtime"] = mtime

        for k, v in pid_to_parts.items():
            if (mtime := v.get("doc_mtime", -1)) == -1:
                logger.warning(f"Duplicate or missing mtime for {k}")
            v["key"] = calculate_doc_key(mtime, v["parts"])
            del v["parts"]

        return pid_to_parts

    def os_client(self):
        return OpenSearchClientWithLogging(**self.os_client_params)

    def delete_os_not_in_source(self, os_pid_to_parts, all_source_ids) -> None:
        with self.os_client() as os:
            # TODO: Split ProcessBatch into just the os writer piece and the other bit, all the
            # nones is a bit weird.
            deleter = self.ProcessBatch(None, None, None, None, None, os, self.target_params)
            for k, v in os_pid_to_parts.items():
                if k in all_source_ids:
                    continue
                for id in v["os_ids"]:
                    self.stats.only_in_os += 1
                    deleter.records.append(
                        OpenSearchDeleteRecord(
                            _index=self.target_params.index_name,
                            _id=id,
                        )
                    )
                    if len(deleter.records) >= deleter.os_batch_size:
                        deleter.write_os_records(None)
            deleter.flush_records(None)

    class ProcessBatch:
        def __init__(self, root, splitter, ids, fid_to_mtime, to_be_deleted, os_client, target_params):
            self.root = root
            self.splitter = splitter
            self.ids = ids
            self.fid_to_mtime = fid_to_mtime
            self.to_be_deleted = to_be_deleted
            self.os_client = os_client
            self.target_params = target_params

            self.retry_count = 0
            self.os_batch_size = 100
            self.records = []
            self.pending_successful_write = {}
            self.id_to_parent_id = {}

        def run(self) -> None:
            mat_dir = _MatDir(self.root)

            # process all deletes first; we can later optimize this to not delete documents that
            # are going to be created, but we only know which ones will be created when we process
            # the files; Ideally we would affect each document in opensearch exactly once.
            # The deletion and re-insertion only happens in failure cases and update cases which
            # are right now expected to be rare, so we can avoid the optimization complexity.
            for i in self.ids:
                if i not in self.to_be_deleted:
                    continue

                for j in self.to_be_deleted[i]:
                    self.records.append(
                        OpenSearchDeleteRecord(
                            _index=self.target_params.index_name,
                            _id=j,
                        )
                    )
                    if len(self.records) >= self.os_batch_size:
                        self.write_os_records(None)

            # Make sure to flush all deletions before starting insertion to avoid
            # race conditions between delete and insert.
            self.flush_records(None)

            for i in self.ids:
                fn = raw_id_to_filename(i)
                doc = mat_dir.read_document(fn)

                self.records.extend(self.split_doc(doc, i))
                if len(self.records) >= self.os_batch_size:
                    self.write_os_records(mat_dir)

            self.flush_records(mat_dir)

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
            assert "doc_mtime" not in parts[0].data
            parts[0].data["doc_mtime"] = self.fid_to_mtime[expected_raw_id]
            ret = []
            if parts[0].parent_id is None:
                assert (
                    parts[0].doc_id == parent_id
                ), f"If first doc has no parent id, it should still have doc_id as its id, but {parts[0].doc_id} != {parent_id}"
                psw[short_doc_id] = True
                ret.append(doc_to_record(parts[0], self.target_params))
                parts = parts[1:]

            for i, p in enumerate(parts):
                assert (
                    p.parent_id == parent_id
                ), f"Subdocs should have proper parent id. got {p.parent_id}; want {parent_id}"
                p.doc_id = "X"
                r = doc_to_record(p, self.target_params)
                assert r._id == "X" and r._source["doc_id"] == "X"
                h = hashlib.sha256(f"{parent_id}/{i}/".encode("UTF-8"))
                # sort keys so the hash should be deterministic.
                h.update(json.dumps(r._source, sort_keys=True).encode("UTF-8"))

                sid = base64.urlsafe_b64encode(h.digest()).decode("UTF-8")
                r._id = "splitdoc-" + sid
                r._source["doc_id"] = r._id
                # See debugging details on key calculation for the drop_subdoc test
                if True and "4e07" in parent_id and i == 0:
                    print(f"DROP_SUBDOC_TEST {parent_id}/{i}/{json.dumps(r._source, sort_keys=True)}")

                ret.append(r)
                psw[sid] = True
                self.id_to_parent_id[sid] = short_doc_id

            psw["key"] = calculate_doc_key(self.fid_to_mtime[expected_raw_id], list(psw.keys()))
            return ret

        def write_os_records(self, mat_dir) -> None:
            def generate_records(records):
                for r in records:
                    assert is_dataclass(r)
                    yield asdict(r)

            unique_id_check = set()
            for r in self.records:
                assert r._id not in unique_id_check
                unique_id_check.add(r._id)

            retry_ids = set()
            for success, item in self.os_client.parallel_bulk(
                generate_records(self.records), **self.target_params.insert_settings
            ):
                if success:
                    if "index" in item:
                        self.handle_index_success(item, mat_dir)
                    elif "delete" in item:
                        pass  # nothing to do
                    else:
                        assert False, f"invalid response from opensearch {item}"
                elif item["index"]["status"] == 429:
                    assert len(item) == 1, f"Fail {item}"
                    retry_ids.add(next(iter(item.values()))["_id"])
                    # opensearch_writer.py uses item["index"]["data"], but from
                    # https://github.com/opensearch-project/opensearch-py/blob/5f6cc2e0072214c8b67c3570598318f7cd73ca9e/opensearchpy/helpers/actions.py#L192
                    # that only seems to exist if raise_on_error is set in which case we
                    # crash, or if the entire chunk failed with a TransportError and so goes
                    # through https://github.com/opensearch-project/opensearch-py/blob/5f6cc2e0072214c8b67c3570598318f7cd73ca9e/opensearchpy/helpers/actions.py#L224
                    # So instead we handle this ourselves since it doesn't look like we can rely
                    # on data always being present.  Worse for deletes we need to get back to the
                    # original value and we can't since delete entries don't have data
                else:
                    msg = f"Failed to upload documnet: {item}"
                    logger.error(msg)
                    raise Exception(msg)

            if len(retry_ids) == 0:
                self.retry_count = 0
                self.records = []
            else:
                self.backoff()
                retry_records = []
                for r in self.records:
                    if r._id in retry_ids:
                        retry_records.append(r)
                self.records = retry_records

        def handle_index_success(self, item, mat_dir) -> None:
            assert mat_dir is not None
            did = item["index"]["_id"]
            if did.startswith("path-sha256-"):
                did = did.removeprefix("path-sha256-")
                assert did in self.pending_successful_write, f"root doc {did} has no entry in pending_successful_write?"
                psw = self.pending_successful_write[did]
                assert did in psw, f"root of docid {did} successfully written, but not a pending successful write?"
                del psw[did]
            elif did.startswith("splitdoc-"):
                cdid = did.removeprefix("splitdoc-")
                assert cdid in self.id_to_parent_id, f"doc id {cdid} successfully written, but not in id_to_parent_id"
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
                path = f"oss-{did}{KEYSEP}{self.fid_to_mtime[did]}{KEYSEP}{psw['key']}.md"
                logger.debug(f"Successfully wrote all parts of {did} touching {path}")
                mat_dir.touch_file(path)

        def backoff(self) -> None:
            if self.retry_count > 6:
                raise Exception("too many consecutive requests that required backoff")

            backoff = 1 * (2**self.retry_count)
            sleep_time = backoff + random.uniform(0, 0.1 * backoff)
            self.retry_count += 1

            logger.warning(f"{self.retry_count} consecutive requests with some 429s. Sleep({sleep_time:.2f}).")
            self.sleep(sleep_time)

        def sleep(self, seconds) -> None:
            time.sleep(seconds)

        def flush_records(self, mat_dir) -> None:
            while len(self.records) > 0:
                self.write_os_records(mat_dir)


@dataclass
class OpenSearchDeleteRecord:
    _index: str
    _id: str
    _op_type: str = "delete"


@dataclass
class SourceFileInfo:
    fid_to_mtime: dict
    id_to_key: dict
    updated_count: int


@dataclass
class SyncStats:
    # Counts of all of these are in documents
    correctly_loaded: int = 0
    missing_md_info: int = 0  # multiple md entries will get classified as this
    updated_source_file: int = 0  # all of these will also be counted in missing_md_info
    missing_os_record: int = 0
    mismatch_key: int = 0
    only_in_os: int = 0
    misformatted_oss_file: int = 0
    ignored_other_md: int = 0
    ignored_unrecognized: int = 0
