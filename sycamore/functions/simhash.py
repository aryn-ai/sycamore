import sys
import operator
from functools import reduce
from sycamore.functions.rabin_karp import RkWindow

__all__ = ["shinglesCalc", "shinglesDist", "simHash", "simHashesDist", "simHashText"]

###############################################################################
#
# Helper Functions
#
###############################################################################


def scramble(val: int) -> int:
    """
    scramble() takes an existing 64-bit hash value and permutes the bits
    into another hash value.  Uses two special constants:
    6364136223846793005 = f-value for Mersenne Twister MT19937-64
    9223372036854775783 = largest prime < 2^63
    """

    return ((val * 6364136223846793005) + 9223372036854775783) & 0xFFFFFFFFFFFFFFFF


def sortedVectorCmp(aVec: list[int], bVec: list[int]) -> tuple[int, int]:
    """
    sortedVectorCmp() takes two sorted lists and compares their elements.
    The tuple returned is (match_count, max_length).
    """

    aLen = len(aVec)
    bLen = len(bVec)
    aIdx = 0
    bIdx = 0
    matches = 0
    while (aIdx < aLen) and (bIdx < bLen):
        aVal = aVec[aIdx]
        bVal = bVec[bIdx]
        if aVal < bVal:
            aIdx += 1
        elif bVal < aVal:
            bIdx += 1
        else:
            matches += 1
            aIdx += 1
            bIdx += 1
    return (matches, max(aLen, bLen))


###############################################################################
#
# Shingles Functions
#
# The analogy here is to shingles on a house.  They overlap each other.
# Conceptually, a set of shingles looks like this:
#
# +--+
# |  |
# +--+
#  |  |
#  +--+   = shingles, with "number" = 4
#   |  |
#   +--+
#    |  |
#    +--+
#
# Each box above represents a single hash.  The hashes are made using a
# sliding window.  For example, if the "window" is 5:
#
#   The quick brown fox
#   |___|=>0x1c91
#    |___|=>0xbe8f
#     |___|=>0x9bed
#
# This code uses a hash function that is efficient for sliding windows
# (Rabin-Karp).  The end result is a list of hashes, with length proportional
# to the size of the input.  They are sorted and the first N constitute a
# set of shingles.
#
###############################################################################


def shinglesCalc(text: bytes, window: int = 17, number: int = 16) -> list[int]:
    """
    shinglesCalc() will process `text` and return a list of hashes.
    This list is often referred to as "shingles" and consists of the
    lowest-value `number` hashes.  Parameter `window` is the number of
    bytes in the sliding window that's hashed.

    text    - The text to process, in UTF-8 bytes
    window  - Width in bytes of the sliding window used for shingles
    number  - The number of least-value shingles to retain
    """

    seen = set()
    ww = RkWindow(window)

    for x in text:
        hh = ww.hash(x)
        if hh is not None:
            seen.add(scramble(hh))

    ary = sorted(seen)
    nn = len(ary)
    if nn == 0:
        return [0] * number
    elif nn < number:
        copies = (number + nn - 1) // nn
        ary *= copies
        ary.sort()
    return ary[:number]


def shinglesDist(aa: list[int], bb: list[int]) -> float:
    """
    shinglesDist() is a distance function for two sets of shingles.
    The outputs of shinglesCalc() can be used here.  The return value
    is a real number [0, 1.0] indicating dissimilarity.
    """

    numer, denom = sortedVectorCmp(aa, bb)
    if denom == 0:
        return 1.0
    return (denom - numer) / denom


###############################################################################
#
# SimHash Functions
#
###############################################################################


def simHash(tab: list[int]) -> int:
    """
    simHash() takes one shingle variant (or "tab"), that is, a vector of
    hashes, and returns a single similarity hash as per Moses Charikar's
    2002 paper.  64-bit is assumed.  For proper results, the number of
    elements in the list should be odd, otherwise the bit distribution
    will be skewed.
    """

    nn = len(tab)
    half = nn // 2
    bit = 0x8000000000000000  # 2^63
    rv = 0
    while bit:
        cnt = 0
        for x in tab:
            if x & bit:
                cnt += 1
        if cnt > half:
            rv |= bit
        bit >>= 1
    return rv


def simHashesDistFast(aa: list[int], bb: list[int]) -> float:
    """
    simHashesDistFast() compares two lists of SimHashes and returns a
    distance metric.  Each list of SimHashes represents a document.
    Corresponding elements in each list represent variants of
    shingles.  With a SimHash, the most bits in common means the most
    similar.  This returns the average of the count of differing bits / 64.
    This fast version for Python >=3.10 takes 50% less time than the slow.
    """

    nn = len(aa)
    assert len(bb) == nn
    tot = 0
    for a, b in zip(aa, bb):
        x = a ^ b
        tot += x.bit_count()  # type: ignore[attr-defined]
    return (tot / nn) / 64.0


def simHashesDistSlow(aa: list[int], bb: list[int]) -> float:
    """
    simHashesDistSlow() compares two lists of SimHashes and returns a
    distance metric.  Each list of SimHashes represents a document.
    Corresponding elements in each list represent variants of
    shingles.  With a SimHash, the most bits in common means the most
    similar.  This returns the average of the count of differing bits / 64.
    This slow version for Python <=3.9 takes 50% more time than the fast.
    """

    nn = len(aa)
    assert len(bb) == nn
    tot = 0
    for a, b in zip(aa, bb):
        x = a ^ b
        tot += bin(x).count("1")  # slow way to count set bits
    return (tot / nn) / 64.0


# Python lacks int.bit_count() until version 3.10
if (sys.version_info.major < 3) or ((sys.version_info.major == 3) and (sys.version_info.minor < 10)):
    simHashesDist = simHashesDistSlow
else:
    simHashesDist = simHashesDistFast


def simHashText(text: bytes, window: int = 17, number: int = 16) -> list[int]:
    """
    Takes text and returns a list of SimHashes.  Arguments:

    text    - The text to process, in UTF-8 bytes
    window  - Width in bytes of the sliding window used for shingles
    number  - The number of variant SimHashes to generate
    """

    ww = RkWindow(window)
    countVecVec = [[0] * 64 for t in range(number)]

    for x in text:
        hh = ww.hash(x)
        if hh is not None:
            for countVec in countVecVec:
                hh = scramble(hh)
                # use lookup table, list appending, and element-wise addition
                countVec[:] = map(
                    operator.add,
                    countVec,
                    simTbl[hh & 0xFF]
                    + simTbl[(hh >> 8) & 0xFF]
                    + simTbl[(hh >> 16) & 0xFF]
                    + simTbl[(hh >> 24) & 0xFF]
                    + simTbl[(hh >> 32) & 0xFF]
                    + simTbl[(hh >> 40) & 0xFF]
                    + simTbl[(hh >> 48) & 0xFF]
                    + simTbl[(hh >> 56) & 0xFF],
                )

    sims = []
    if len(text) < window:
        hh = ww.get()
        for i in range(number):
            hh = scramble(hh)  # type: ignore[arg-type]
            sims.append(hh)
        return sims

    for countVec in countVecVec:
        # convert array of 64 counts to binary string, to int, and append
        sims.append(int(reduce(lambda s, x: ("1" if x >= 0 else "0") + s, countVec, ""), 2))
    return sims


# Lookup table to speed up SimHash calculation.  For each byte, for each bit,
# value is 1 if bit is set, -1 if not set.  Bit 0 is least-significant.
# fmt: off
simTbl = [
    [-1, -1, -1, -1, -1, -1, -1, -1],  # 00
    [ 1, -1, -1, -1, -1, -1, -1, -1],  # 01
    [-1,  1, -1, -1, -1, -1, -1, -1],  # 02
    [ 1,  1, -1, -1, -1, -1, -1, -1],  # 03
    [-1, -1,  1, -1, -1, -1, -1, -1],  # 04
    [ 1, -1,  1, -1, -1, -1, -1, -1],  # 05
    [-1,  1,  1, -1, -1, -1, -1, -1],  # 06
    [ 1,  1,  1, -1, -1, -1, -1, -1],  # 07
    [-1, -1, -1,  1, -1, -1, -1, -1],  # 08
    [ 1, -1, -1,  1, -1, -1, -1, -1],  # 09
    [-1,  1, -1,  1, -1, -1, -1, -1],  # 0a
    [ 1,  1, -1,  1, -1, -1, -1, -1],  # 0b
    [-1, -1,  1,  1, -1, -1, -1, -1],  # 0c
    [ 1, -1,  1,  1, -1, -1, -1, -1],  # 0d
    [-1,  1,  1,  1, -1, -1, -1, -1],  # 0e
    [ 1,  1,  1,  1, -1, -1, -1, -1],  # 0f
    [-1, -1, -1, -1,  1, -1, -1, -1],  # 10
    [ 1, -1, -1, -1,  1, -1, -1, -1],  # 11
    [-1,  1, -1, -1,  1, -1, -1, -1],  # 12
    [ 1,  1, -1, -1,  1, -1, -1, -1],  # 13
    [-1, -1,  1, -1,  1, -1, -1, -1],  # 14
    [ 1, -1,  1, -1,  1, -1, -1, -1],  # 15
    [-1,  1,  1, -1,  1, -1, -1, -1],  # 16
    [ 1,  1,  1, -1,  1, -1, -1, -1],  # 17
    [-1, -1, -1,  1,  1, -1, -1, -1],  # 18
    [ 1, -1, -1,  1,  1, -1, -1, -1],  # 19
    [-1,  1, -1,  1,  1, -1, -1, -1],  # 1a
    [ 1,  1, -1,  1,  1, -1, -1, -1],  # 1b
    [-1, -1,  1,  1,  1, -1, -1, -1],  # 1c
    [ 1, -1,  1,  1,  1, -1, -1, -1],  # 1d
    [-1,  1,  1,  1,  1, -1, -1, -1],  # 1e
    [ 1,  1,  1,  1,  1, -1, -1, -1],  # 1f
    [-1, -1, -1, -1, -1,  1, -1, -1],  # 20
    [ 1, -1, -1, -1, -1,  1, -1, -1],  # 21
    [-1,  1, -1, -1, -1,  1, -1, -1],  # 22
    [ 1,  1, -1, -1, -1,  1, -1, -1],  # 23
    [-1, -1,  1, -1, -1,  1, -1, -1],  # 24
    [ 1, -1,  1, -1, -1,  1, -1, -1],  # 25
    [-1,  1,  1, -1, -1,  1, -1, -1],  # 26
    [ 1,  1,  1, -1, -1,  1, -1, -1],  # 27
    [-1, -1, -1,  1, -1,  1, -1, -1],  # 28
    [ 1, -1, -1,  1, -1,  1, -1, -1],  # 29
    [-1,  1, -1,  1, -1,  1, -1, -1],  # 2a
    [ 1,  1, -1,  1, -1,  1, -1, -1],  # 2b
    [-1, -1,  1,  1, -1,  1, -1, -1],  # 2c
    [ 1, -1,  1,  1, -1,  1, -1, -1],  # 2d
    [-1,  1,  1,  1, -1,  1, -1, -1],  # 2e
    [ 1,  1,  1,  1, -1,  1, -1, -1],  # 2f
    [-1, -1, -1, -1,  1,  1, -1, -1],  # 30
    [ 1, -1, -1, -1,  1,  1, -1, -1],  # 31
    [-1,  1, -1, -1,  1,  1, -1, -1],  # 32
    [ 1,  1, -1, -1,  1,  1, -1, -1],  # 33
    [-1, -1,  1, -1,  1,  1, -1, -1],  # 34
    [ 1, -1,  1, -1,  1,  1, -1, -1],  # 35
    [-1,  1,  1, -1,  1,  1, -1, -1],  # 36
    [ 1,  1,  1, -1,  1,  1, -1, -1],  # 37
    [-1, -1, -1,  1,  1,  1, -1, -1],  # 38
    [ 1, -1, -1,  1,  1,  1, -1, -1],  # 39
    [-1,  1, -1,  1,  1,  1, -1, -1],  # 3a
    [ 1,  1, -1,  1,  1,  1, -1, -1],  # 3b
    [-1, -1,  1,  1,  1,  1, -1, -1],  # 3c
    [ 1, -1,  1,  1,  1,  1, -1, -1],  # 3d
    [-1,  1,  1,  1,  1,  1, -1, -1],  # 3e
    [ 1,  1,  1,  1,  1,  1, -1, -1],  # 3f
    [-1, -1, -1, -1, -1, -1,  1, -1],  # 40
    [ 1, -1, -1, -1, -1, -1,  1, -1],  # 41
    [-1,  1, -1, -1, -1, -1,  1, -1],  # 42
    [ 1,  1, -1, -1, -1, -1,  1, -1],  # 43
    [-1, -1,  1, -1, -1, -1,  1, -1],  # 44
    [ 1, -1,  1, -1, -1, -1,  1, -1],  # 45
    [-1,  1,  1, -1, -1, -1,  1, -1],  # 46
    [ 1,  1,  1, -1, -1, -1,  1, -1],  # 47
    [-1, -1, -1,  1, -1, -1,  1, -1],  # 48
    [ 1, -1, -1,  1, -1, -1,  1, -1],  # 49
    [-1,  1, -1,  1, -1, -1,  1, -1],  # 4a
    [ 1,  1, -1,  1, -1, -1,  1, -1],  # 4b
    [-1, -1,  1,  1, -1, -1,  1, -1],  # 4c
    [ 1, -1,  1,  1, -1, -1,  1, -1],  # 4d
    [-1,  1,  1,  1, -1, -1,  1, -1],  # 4e
    [ 1,  1,  1,  1, -1, -1,  1, -1],  # 4f
    [-1, -1, -1, -1,  1, -1,  1, -1],  # 50
    [ 1, -1, -1, -1,  1, -1,  1, -1],  # 51
    [-1,  1, -1, -1,  1, -1,  1, -1],  # 52
    [ 1,  1, -1, -1,  1, -1,  1, -1],  # 53
    [-1, -1,  1, -1,  1, -1,  1, -1],  # 54
    [ 1, -1,  1, -1,  1, -1,  1, -1],  # 55
    [-1,  1,  1, -1,  1, -1,  1, -1],  # 56
    [ 1,  1,  1, -1,  1, -1,  1, -1],  # 57
    [-1, -1, -1,  1,  1, -1,  1, -1],  # 58
    [ 1, -1, -1,  1,  1, -1,  1, -1],  # 59
    [-1,  1, -1,  1,  1, -1,  1, -1],  # 5a
    [ 1,  1, -1,  1,  1, -1,  1, -1],  # 5b
    [-1, -1,  1,  1,  1, -1,  1, -1],  # 5c
    [ 1, -1,  1,  1,  1, -1,  1, -1],  # 5d
    [-1,  1,  1,  1,  1, -1,  1, -1],  # 5e
    [ 1,  1,  1,  1,  1, -1,  1, -1],  # 5f
    [-1, -1, -1, -1, -1,  1,  1, -1],  # 60
    [ 1, -1, -1, -1, -1,  1,  1, -1],  # 61
    [-1,  1, -1, -1, -1,  1,  1, -1],  # 62
    [ 1,  1, -1, -1, -1,  1,  1, -1],  # 63
    [-1, -1,  1, -1, -1,  1,  1, -1],  # 64
    [ 1, -1,  1, -1, -1,  1,  1, -1],  # 65
    [-1,  1,  1, -1, -1,  1,  1, -1],  # 66
    [ 1,  1,  1, -1, -1,  1,  1, -1],  # 67
    [-1, -1, -1,  1, -1,  1,  1, -1],  # 68
    [ 1, -1, -1,  1, -1,  1,  1, -1],  # 69
    [-1,  1, -1,  1, -1,  1,  1, -1],  # 6a
    [ 1,  1, -1,  1, -1,  1,  1, -1],  # 6b
    [-1, -1,  1,  1, -1,  1,  1, -1],  # 6c
    [ 1, -1,  1,  1, -1,  1,  1, -1],  # 6d
    [-1,  1,  1,  1, -1,  1,  1, -1],  # 6e
    [ 1,  1,  1,  1, -1,  1,  1, -1],  # 6f
    [-1, -1, -1, -1,  1,  1,  1, -1],  # 70
    [ 1, -1, -1, -1,  1,  1,  1, -1],  # 71
    [-1,  1, -1, -1,  1,  1,  1, -1],  # 72
    [ 1,  1, -1, -1,  1,  1,  1, -1],  # 73
    [-1, -1,  1, -1,  1,  1,  1, -1],  # 74
    [ 1, -1,  1, -1,  1,  1,  1, -1],  # 75
    [-1,  1,  1, -1,  1,  1,  1, -1],  # 76
    [ 1,  1,  1, -1,  1,  1,  1, -1],  # 77
    [-1, -1, -1,  1,  1,  1,  1, -1],  # 78
    [ 1, -1, -1,  1,  1,  1,  1, -1],  # 79
    [-1,  1, -1,  1,  1,  1,  1, -1],  # 7a
    [ 1,  1, -1,  1,  1,  1,  1, -1],  # 7b
    [-1, -1,  1,  1,  1,  1,  1, -1],  # 7c
    [ 1, -1,  1,  1,  1,  1,  1, -1],  # 7d
    [-1,  1,  1,  1,  1,  1,  1, -1],  # 7e
    [ 1,  1,  1,  1,  1,  1,  1, -1],  # 7f
    [-1, -1, -1, -1, -1, -1, -1,  1],  # 80
    [ 1, -1, -1, -1, -1, -1, -1,  1],  # 81
    [-1,  1, -1, -1, -1, -1, -1,  1],  # 82
    [ 1,  1, -1, -1, -1, -1, -1,  1],  # 83
    [-1, -1,  1, -1, -1, -1, -1,  1],  # 84
    [ 1, -1,  1, -1, -1, -1, -1,  1],  # 85
    [-1,  1,  1, -1, -1, -1, -1,  1],  # 86
    [ 1,  1,  1, -1, -1, -1, -1,  1],  # 87
    [-1, -1, -1,  1, -1, -1, -1,  1],  # 88
    [ 1, -1, -1,  1, -1, -1, -1,  1],  # 89
    [-1,  1, -1,  1, -1, -1, -1,  1],  # 8a
    [ 1,  1, -1,  1, -1, -1, -1,  1],  # 8b
    [-1, -1,  1,  1, -1, -1, -1,  1],  # 8c
    [ 1, -1,  1,  1, -1, -1, -1,  1],  # 8d
    [-1,  1,  1,  1, -1, -1, -1,  1],  # 8e
    [ 1,  1,  1,  1, -1, -1, -1,  1],  # 8f
    [-1, -1, -1, -1,  1, -1, -1,  1],  # 90
    [ 1, -1, -1, -1,  1, -1, -1,  1],  # 91
    [-1,  1, -1, -1,  1, -1, -1,  1],  # 92
    [ 1,  1, -1, -1,  1, -1, -1,  1],  # 93
    [-1, -1,  1, -1,  1, -1, -1,  1],  # 94
    [ 1, -1,  1, -1,  1, -1, -1,  1],  # 95
    [-1,  1,  1, -1,  1, -1, -1,  1],  # 96
    [ 1,  1,  1, -1,  1, -1, -1,  1],  # 97
    [-1, -1, -1,  1,  1, -1, -1,  1],  # 98
    [ 1, -1, -1,  1,  1, -1, -1,  1],  # 99
    [-1,  1, -1,  1,  1, -1, -1,  1],  # 9a
    [ 1,  1, -1,  1,  1, -1, -1,  1],  # 9b
    [-1, -1,  1,  1,  1, -1, -1,  1],  # 9c
    [ 1, -1,  1,  1,  1, -1, -1,  1],  # 9d
    [-1,  1,  1,  1,  1, -1, -1,  1],  # 9e
    [ 1,  1,  1,  1,  1, -1, -1,  1],  # 9f
    [-1, -1, -1, -1, -1,  1, -1,  1],  # a0
    [ 1, -1, -1, -1, -1,  1, -1,  1],  # a1
    [-1,  1, -1, -1, -1,  1, -1,  1],  # a2
    [ 1,  1, -1, -1, -1,  1, -1,  1],  # a3
    [-1, -1,  1, -1, -1,  1, -1,  1],  # a4
    [ 1, -1,  1, -1, -1,  1, -1,  1],  # a5
    [-1,  1,  1, -1, -1,  1, -1,  1],  # a6
    [ 1,  1,  1, -1, -1,  1, -1,  1],  # a7
    [-1, -1, -1,  1, -1,  1, -1,  1],  # a8
    [ 1, -1, -1,  1, -1,  1, -1,  1],  # a9
    [-1,  1, -1,  1, -1,  1, -1,  1],  # aa
    [ 1,  1, -1,  1, -1,  1, -1,  1],  # ab
    [-1, -1,  1,  1, -1,  1, -1,  1],  # ac
    [ 1, -1,  1,  1, -1,  1, -1,  1],  # ad
    [-1,  1,  1,  1, -1,  1, -1,  1],  # ae
    [ 1,  1,  1,  1, -1,  1, -1,  1],  # af
    [-1, -1, -1, -1,  1,  1, -1,  1],  # b0
    [ 1, -1, -1, -1,  1,  1, -1,  1],  # b1
    [-1,  1, -1, -1,  1,  1, -1,  1],  # b2
    [ 1,  1, -1, -1,  1,  1, -1,  1],  # b3
    [-1, -1,  1, -1,  1,  1, -1,  1],  # b4
    [ 1, -1,  1, -1,  1,  1, -1,  1],  # b5
    [-1,  1,  1, -1,  1,  1, -1,  1],  # b6
    [ 1,  1,  1, -1,  1,  1, -1,  1],  # b7
    [-1, -1, -1,  1,  1,  1, -1,  1],  # b8
    [ 1, -1, -1,  1,  1,  1, -1,  1],  # b9
    [-1,  1, -1,  1,  1,  1, -1,  1],  # ba
    [ 1,  1, -1,  1,  1,  1, -1,  1],  # bb
    [-1, -1,  1,  1,  1,  1, -1,  1],  # bc
    [ 1, -1,  1,  1,  1,  1, -1,  1],  # bd
    [-1,  1,  1,  1,  1,  1, -1,  1],  # be
    [ 1,  1,  1,  1,  1,  1, -1,  1],  # bf
    [-1, -1, -1, -1, -1, -1,  1,  1],  # c0
    [ 1, -1, -1, -1, -1, -1,  1,  1],  # c1
    [-1,  1, -1, -1, -1, -1,  1,  1],  # c2
    [ 1,  1, -1, -1, -1, -1,  1,  1],  # c3
    [-1, -1,  1, -1, -1, -1,  1,  1],  # c4
    [ 1, -1,  1, -1, -1, -1,  1,  1],  # c5
    [-1,  1,  1, -1, -1, -1,  1,  1],  # c6
    [ 1,  1,  1, -1, -1, -1,  1,  1],  # c7
    [-1, -1, -1,  1, -1, -1,  1,  1],  # c8
    [ 1, -1, -1,  1, -1, -1,  1,  1],  # c9
    [-1,  1, -1,  1, -1, -1,  1,  1],  # ca
    [ 1,  1, -1,  1, -1, -1,  1,  1],  # cb
    [-1, -1,  1,  1, -1, -1,  1,  1],  # cc
    [ 1, -1,  1,  1, -1, -1,  1,  1],  # cd
    [-1,  1,  1,  1, -1, -1,  1,  1],  # ce
    [ 1,  1,  1,  1, -1, -1,  1,  1],  # cf
    [-1, -1, -1, -1,  1, -1,  1,  1],  # d0
    [ 1, -1, -1, -1,  1, -1,  1,  1],  # d1
    [-1,  1, -1, -1,  1, -1,  1,  1],  # d2
    [ 1,  1, -1, -1,  1, -1,  1,  1],  # d3
    [-1, -1,  1, -1,  1, -1,  1,  1],  # d4
    [ 1, -1,  1, -1,  1, -1,  1,  1],  # d5
    [-1,  1,  1, -1,  1, -1,  1,  1],  # d6
    [ 1,  1,  1, -1,  1, -1,  1,  1],  # d7
    [-1, -1, -1,  1,  1, -1,  1,  1],  # d8
    [ 1, -1, -1,  1,  1, -1,  1,  1],  # d9
    [-1,  1, -1,  1,  1, -1,  1,  1],  # da
    [ 1,  1, -1,  1,  1, -1,  1,  1],  # db
    [-1, -1,  1,  1,  1, -1,  1,  1],  # dc
    [ 1, -1,  1,  1,  1, -1,  1,  1],  # dd
    [-1,  1,  1,  1,  1, -1,  1,  1],  # de
    [ 1,  1,  1,  1,  1, -1,  1,  1],  # df
    [-1, -1, -1, -1, -1,  1,  1,  1],  # e0
    [ 1, -1, -1, -1, -1,  1,  1,  1],  # e1
    [-1,  1, -1, -1, -1,  1,  1,  1],  # e2
    [ 1,  1, -1, -1, -1,  1,  1,  1],  # e3
    [-1, -1,  1, -1, -1,  1,  1,  1],  # e4
    [ 1, -1,  1, -1, -1,  1,  1,  1],  # e5
    [-1,  1,  1, -1, -1,  1,  1,  1],  # e6
    [ 1,  1,  1, -1, -1,  1,  1,  1],  # e7
    [-1, -1, -1,  1, -1,  1,  1,  1],  # e8
    [ 1, -1, -1,  1, -1,  1,  1,  1],  # e9
    [-1,  1, -1,  1, -1,  1,  1,  1],  # ea
    [ 1,  1, -1,  1, -1,  1,  1,  1],  # eb
    [-1, -1,  1,  1, -1,  1,  1,  1],  # ec
    [ 1, -1,  1,  1, -1,  1,  1,  1],  # ed
    [-1,  1,  1,  1, -1,  1,  1,  1],  # ee
    [ 1,  1,  1,  1, -1,  1,  1,  1],  # ef
    [-1, -1, -1, -1,  1,  1,  1,  1],  # f0
    [ 1, -1, -1, -1,  1,  1,  1,  1],  # f1
    [-1,  1, -1, -1,  1,  1,  1,  1],  # f2
    [ 1,  1, -1, -1,  1,  1,  1,  1],  # f3
    [-1, -1,  1, -1,  1,  1,  1,  1],  # f4
    [ 1, -1,  1, -1,  1,  1,  1,  1],  # f5
    [-1,  1,  1, -1,  1,  1,  1,  1],  # f6
    [ 1,  1,  1, -1,  1,  1,  1,  1],  # f7
    [-1, -1, -1,  1,  1,  1,  1,  1],  # f8
    [ 1, -1, -1,  1,  1,  1,  1,  1],  # f9
    [-1,  1, -1,  1,  1,  1,  1,  1],  # fa
    [ 1,  1, -1,  1,  1,  1,  1,  1],  # fb
    [-1, -1,  1,  1,  1,  1,  1,  1],  # fc
    [ 1, -1,  1,  1,  1,  1,  1,  1],  # fd
    [-1,  1,  1,  1,  1,  1,  1,  1],  # fe
    [ 1,  1,  1,  1,  1,  1,  1,  1],  # ff
]
# fmt: on
