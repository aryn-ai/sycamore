from typing import Optional

import nanoid

# Alphabets for encodings...
alpha36 = "0123456789abcdefghijklmnopqrstuvwxyz"
alpha16 = "0123456789abcdef"

docid_nanoid_chars = 23  # 36^23 is a bit less than 2^119 (~15 bytes)


def nanoid36() -> str:
    """
    Free of punctuation and uppercase; still as good as UUID4.
    """
    return nanoid.generate(alpha36, docid_nanoid_chars)


def mkdocid(code: str = "d") -> str:
    """
    Docid that qualifies as a URI with aryn: scheme.
    """
    return f"aryn:{code}-{nanoid36()}"


def docid_to_uuid(id: Optional[str]) -> Optional[str]:
    if not id or not id.startswith("aryn:"):
        return id
    left, right = id.split("-", 1)
    return nanoid36_to_uuid(right)


def uuid_to_docid(uu: Optional[str], code: str = "d") -> Optional[str]:
    if not uu:
        return uu
    id = uuid_to_nanoid36(uu)
    return f"aryn:{code}-{id}"


def nanoid36_to_uuid(id: str) -> str:
    """
    Invertable conversion of docid to UUID for application that need UUID.
    See RFC 9562 for details.
    """
    x = str_to_bignum(id, alpha36)
    nybbles = bignum_to_nybbles(x, 30)  # 30 is ceil(119 / 4); nybble is 4 bits
    nybbles[12:12] = [4]  # insert indicator of version 4
    nybbles[16:16] = [8]  # insert indicator of variant OSF DCE
    return nybbles_to_uuid(nybbles)


def uuid_to_nanoid36(uu: str) -> str:
    """
    Reverse operation for docid_to_uuid().
    """
    nybbles = str_to_list(uu, alpha16)
    nybbles[16:17] = []  # remove variant
    nybbles[12:13] = []  # remove version
    x = nybbles_to_bignum(nybbles)
    return bignum_to_str(x, alpha36, docid_nanoid_chars)


def str_to_bignum(s: str, alpha: str) -> int:
    n = len(alpha)
    rv = 0
    for ch in s:
        idx = alpha.index(ch)
        rv = (rv * n) + idx
    return rv


def bignum_to_str(x: int, alpha: str, chars: int) -> str:
    n = len(alpha)
    rv = ""
    for _ in range(chars):
        x, rem = divmod(x, n)
        ch = alpha[rem]
        rv = ch + rv  # prepend
    return rv


def str_to_list(s: str, alpha: str) -> list[int]:
    rv = []
    for ch in s:
        if (i := alpha.find(ch)) >= 0:
            rv.append(i)
    return rv


def bignum_to_nybbles(x: int, count: int) -> list[int]:
    rv = []
    for _ in range(count):
        x, rem = divmod(x, 16)  # a nybble is 4 bits (half a byte)
        rv.append(rem)
    return rv


def nybbles_to_bignum(nybbles: list[int]) -> int:
    rv = 0
    for nybble in reversed(nybbles):
        rv = (rv * 16) + nybble
    return rv


def nybbles_to_uuid(nybbles: list[int]) -> str:
    rv = ""
    for i, nybble in enumerate(nybbles):
        if i in (8, 12, 16, 20):
            rv += "-"
        rv += alpha16[nybble]
    return rv
