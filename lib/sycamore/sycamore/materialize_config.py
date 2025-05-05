from enum import Enum
from abc import ABC, abstractmethod
from sycamore.data import Document, MetadataDocument
from pathlib import Path
import logging
from typing import Optional

from sycamore.data.docid import mkdocid, path_to_sha256_docid, docid_to_typed_nanoid, typed_nanoid_to_docid


class MaterializeSourceMode(Enum):
    """
    See DocSet.materialize for documentation on the semantics of these
    """

    RECOMPUTE = 0
    USE_STORED = 1

    # Deprecated constants
    OFF = 0
    IF_PRESENT = 1


class MaterializeNameStability(Enum):
    RANDOM = 1
    DOC_PATH = 2
    DOC_CONTENT = 3


class MaterializeNameGroup(ABC):
    """
    Base class for groups of functions to convert between Documents, DocIds,
    names of materialize pickle files, and file paths. It should be possible
    to reliably and symmetrically translate between these things.
    """

    @classmethod
    @abstractmethod
    def materialize_name_to_docid(cls, mname: str) -> str:
        pass

    @classmethod
    def materialize_name_to_docid_safe(cls, mname: str) -> Optional[str]:
        try:
            return cls.materialize_name_to_docid(mname)
        except Exception:
            return None

    @classmethod
    @abstractmethod
    def make_docid(cls, doc: Document) -> str:
        pass

    @classmethod
    @abstractmethod
    def doc_to_materialize_name(cls, doc: Document, bin: bytes) -> str:
        pass

    @classmethod
    @abstractmethod
    def docpath_to_docid(cls, docpath: str) -> str:
        pass

    @classmethod
    def is_metadata_materialize_name(cls, mname: str) -> bool:
        return Path(mname).name.startswith("md-")

    @classmethod
    @abstractmethod
    def stability(cls) -> MaterializeNameStability:
        pass


class MRRNameGroup(MaterializeNameGroup):
    """
    MaterializeNameGroup used by MaterializeReadReliability (MRR) to make sure
    document filepaths always give the same docid. MetadataDocument docids fall
    back to RandomNameGroup (sycamore default)
    """

    @classmethod
    def materialize_name_to_docid(cls, mname: str) -> str:
        p = Path(mname)
        assert p.suffix == ".pickle", f"Expected .pickle file, got {p.suffix}"
        if cls.is_metadata_materialize_name(p.name):
            return RandomNameGroup.materialize_name_to_docid(mname)
        assert p.name.startswith(
            "doc-path-sha256-"
        ), "Got pickle file which is not in 'doc-path-sha256-' format with MRR naming"
        return str(p.stem[4:])

    @classmethod
    def make_docid(cls, doc: Document) -> str:
        assert "path" in doc.properties, "Need path property in order to make doc id with MRR naming"
        did = path_to_sha256_docid(doc.properties["path"])
        doc.doc_id = did
        return did

    @classmethod
    def doc_to_materialize_name(cls, doc: Document, bin: bytes) -> str:
        did = doc.doc_id
        if isinstance(doc, MetadataDocument):
            return RandomNameGroup.doc_to_materialize_name(doc, bin)
        assert did is not None, "MRR naming requires a sycamore doc_id, which is missing"
        assert (
            len(did) == 76
        ), f"""This method expects docids to be 76 characters long and used with reliability.
            Make sure to have docids set using docid_from_path method. Found: {did}"""
        assert did.startswith("path-sha256-"), "Docid is not in 'path-sha256-' format with MRR naming"
        return f"doc-{did}.pickle"

    @classmethod
    def docpath_to_docid(cls, docpath: str) -> str:
        return path_to_sha256_docid(docpath)

    @classmethod
    def stability(cls) -> MaterializeNameStability:
        return MaterializeNameStability.DOC_PATH


class RandomNameGroup(MaterializeNameGroup):
    """
    Default MaterializeNameGroup used by sycamore. DocIds are randomly generated.
    """

    @classmethod
    def materialize_name_to_docid(cls, mname: str) -> str:
        p = Path(mname)
        assert p.suffix == ".pickle", f"Expected .pickle suffix, got {p.suffix}"
        parts = p.name.split(sep=".")
        doc_id_part = parts[0]
        if cls.is_metadata_materialize_name(doc_id_part):
            typed_nanoid = doc_id_part[3:]
        elif doc_id_part.startswith("doc-"):
            typed_nanoid = doc_id_part[4:]
        else:
            raise ValueError(f"Unexpected docid format: {doc_id_part}")
        return typed_nanoid_to_docid(typed_nanoid)

    @classmethod
    def make_docid(cls, doc: Document) -> str:
        did = doc.doc_id
        if did is not None:
            return did
        did = mkdocid()
        doc.doc_id = did
        return did

    @classmethod
    def doc_to_materialize_name(cls, doc: Document, bin: bytes) -> str:
        from hashlib import sha256

        hash_id = sha256(bin).hexdigest()
        doc_id = doc.doc_id or doc.data.get("lineage_id", None)
        if doc_id is None:
            logging.warn(f"found document with no doc_id or lineage_id, assigned content based id {hash_id}")
            doc_id = hash_id

        if isinstance(doc, MetadataDocument):
            return f"md-{docid_to_typed_nanoid(doc_id)}.{hash_id}.pickle"

        assert isinstance(doc, Document)
        return f"doc-{docid_to_typed_nanoid(doc_id)}.{hash_id}.pickle"

    @classmethod
    def docpath_to_docid(cls, docpath: str) -> str:
        raise NotImplementedError(
            "RandomNameGroup naming scheme cannot make reliable doc_ids out of file paths, by design."
        )

    @classmethod
    def stability(cls) -> MaterializeNameStability:
        return MaterializeNameStability.RANDOM
