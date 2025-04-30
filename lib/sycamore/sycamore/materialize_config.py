from enum import Enum
from abc import ABC, abstractmethod
from sycamore.data import Document, MetadataDocument
from pathlib import Path
import logging

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

    @abstractmethod
    @staticmethod
    def materialize_name_to_docid(mname: str) -> str:
        pass

    @abstractmethod
    @staticmethod
    def make_docid(doc: Document) -> str:
        pass

    @abstractmethod
    @staticmethod
    def doc_to_materialize_name(doc: Document, bin: bytes) -> str:
        pass

    @abstractmethod
    @staticmethod
    def docpath_to_docid(docpath: str) -> str:
        pass

    @abstractmethod
    @staticmethod
    def stability() -> MaterializeNameStability:
        pass


class MRRNameGroup(MaterializeNameGroup):
    @staticmethod
    def materialize_name_to_docid(mname: str) -> str:
        p = Path(mname)
        assert p.suffix == ".pickle", f"Expected .pickle file, got {p.suffix}"
        assert p.name.startswith(
            "doc-path-sha256-"
        ), "Got pickle file which is not in 'doc-path-sha256-' format with MRR naming"
        return str(p.stem[4:])

    @staticmethod
    def make_docid(doc: Document) -> str:
        assert "path" in doc.properties, "Need path property in order to make doc id with MRR naming"
        did = path_to_sha256_docid(doc.properties["path"])
        doc.doc_id = did
        return did

    @staticmethod
    def doc_to_materialize_name(doc: Document, bin: bytes) -> str:
        did = doc.doc_id
        assert did is not None, "MRR naming requires a sycamore doc_id, which is missing"
        assert (
            len(did) == 76
        ), """This method expects docids to be 76 characters long and used with reliability.
            Make sure to have docids set using docid_from_path method,"""
        assert did.startswith("path-sha256-"), "Docid is not in 'path-sha256-' format with MRR naming"
        if isinstance(doc, MetadataDocument):
            return f"md-{did}.pickle"
        else:
            return f"doc-{did}.pickle"

    @staticmethod
    def docpath_to_docid(docpath: str) -> str:
        return path_to_sha256_docid(docpath)

    @staticmethod
    def stability() -> MaterializeNameStability:
        return MaterializeNameStability.DOC_PATH


class RandomNameGroup(MaterializeNameGroup):
    @staticmethod
    def materialize_name_to_docid(mname: str) -> str:
        p = Path(mname)
        assert p.suffix == ".pickle", f"Expected .pickle suffix, got {p.suffix}"
        parts = p.name.split(sep=".")
        doc_id_part = parts[0]
        if doc_id_part.startswith("md-"):
            typed_nanoid = doc_id_part[3:]
        elif doc_id_part.startswith("doc-"):
            typed_nanoid = doc_id_part[4:]
        else:
            raise ValueError(f"Unexpected docid format: {doc_id_part}")
        return typed_nanoid_to_docid(typed_nanoid)

    @staticmethod
    def make_docid(doc: Document) -> str:
        did = doc.doc_id
        if did is not None:
            return did
        did = mkdocid()
        doc.doc_id = did
        return did

    @staticmethod
    def doc_to_materialize_name(doc: Document, bin: bytes) -> str:
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

    @staticmethod
    def docpath_to_docid(docpath: str) -> str:
        raise NotImplementedError(
            "RandomNameGroup naming scheme cannot make reliable doc_ids out of file paths, by design."
        )

    @staticmethod
    def stability() -> MaterializeNameStability:
        return MaterializeNameStability.RANDOM
