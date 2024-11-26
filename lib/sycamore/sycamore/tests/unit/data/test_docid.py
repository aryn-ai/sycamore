import re
import uuid

from sycamore.data.docid import (
    alpha16,
    alpha36,
    bignum_to_nybbles,
    bignum_to_str,
    docid_nanoid_chars,
    docid_to_uuid,
    mkdocid,
    nanoid36,
    nybbles_to_bignum,
    nybbles_to_uuid,
    str_to_bignum,
    uuid_to_docid,
)


def test_formats():
    re_nano = "[0-9a-z]{" + str(docid_nanoid_chars) + "}"
    nn = nanoid36()
    assert re.fullmatch(re_nano, nn)
    id = mkdocid()
    re_docid = "aryn:d-" + re_nano
    assert re.fullmatch(re_docid, id)
    re_uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    assert re.fullmatch(re_uuid, docid_to_uuid(id))
    assert re.fullmatch(re_uuid, str(uuid.uuid4()))


def test_convert():
    for _ in range(999):
        id = mkdocid()
        fwd = docid_to_uuid(id)
        rev = uuid_to_docid(fwd)
        assert id == rev


def test_zero():
    uu = "00000000-0000-0000-0000-000000000000"
    id = "aryn:d-00000000000000000000000"
    assert uuid_to_docid(uu) == id
    assert docid_to_uuid(id) == "00000000-0000-4000-8000-000000000000"


def test_noconvert():
    assert docid_to_uuid(None) is None
    assert docid_to_uuid("e6797018-dff6") == "e6797018-dff6"
    assert uuid_to_docid(None) is None


def test_internal():
    assert str_to_bignum("deadbeef", alpha16) == 0xDEADBEEF
    assert str_to_bignum("alexmeyer", alpha36) == 29889254506419
    assert bignum_to_str(29889254506419, alpha36, 9) == "alexmeyer"
    big = 0x0123456789ABCDEF
    nyb = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert bignum_to_nybbles(big, 16) == nyb
    assert nybbles_to_bignum(nyb) == big
    assert nybbles_to_uuid(nyb + nyb) == "fedcba98-7654-3210-fedc-ba9876543210"
