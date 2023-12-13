from functools import reduce


class RkHash:
    """
    Class RkHash encapsulates a 64-bit Rabin-Karp hash function.  It is
    meant to hash one byte [0,255] at a time.  It's not the best hash
    function, but has the nice property of allowing previous inputs to
    be removed.  This makes it efficient for sliding-window applications.
    """

    __slots__ = ["shift", "prime", "primeShift", "val", "inv"]

    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError
        self.shift = 8
        self.prime = 36028797018963913  # largest prime < 2^55
        self.primeShift = self.prime * self.shift
        self.val = 0
        self.inv = reduce(lambda x, _: (x << self.shift) % self.prime, range(width - 1), 1)

    def __str__(self) -> str:
        return "(rk:%d:%d)" % (self.val, self.inv)

    def hashIn(self, ch: int) -> None:
        self.val = ((self.val << self.shift) + ch) % self.prime

    def hashOut(self, ch: int) -> None:
        self.val = (self.val + self.primeShift - (ch * self.inv)) % self.prime

    def get(self) -> int:
        return self.val


class RkWindow:
    """
    Class RkWindow is a higher-level interface to RkHash.  It implements
    the sliding window by remembering the last W inputs and orchestrating
    the hashIn() and hashOut() operations.  Small windows limit the range
    of the resulting hashes.  Aim for a minimum size of 7 or 8.
    """

    __slots__ = ["hasher", "width", "ary", "idx"]

    def __init__(self, width: int) -> None:
        if width < 1:
            raise ValueError
        hasher = RkHash(width)
        ary = []
        seed = self.filler()
        for ii in range(width):
            hasher.hashIn(seed)
            ary.append(seed)
        self.hasher = hasher
        self.ary = ary
        self.idx = 0
        self.width = width

    def __str__(self) -> str:
        h = str(self.hasher)
        a = ""
        for ii, ch in enumerate(self.ary):
            if ii == self.idx:
                a += ",[%d]" % ch
            else:
                a += ",%d" % ch
        return "(w%s%s)" % (h, a)

    def hash(self, ch: int) -> None:
        hasher = self.hasher
        ary = self.ary
        idx = self.idx
        hasher.hashOut(ary[idx])
        hasher.hashIn(ch)
        ary[idx] = ch
        idx += 1
        if idx < self.width:
            self.idx = idx
        else:
            self.idx = 0

    def get(self) -> int:
        return self.hasher.val

    @staticmethod
    def filler():
        return 149  # 8-bit prime with good distribution of bits

###############################################################################

# 1-based max-heap
def downHeap(heap, idx):
    """
    downHeap() implements the down-heap operation on a binary max-heap
    represented as a 1-based list.  If the list represents a valid heap
    except that the element at idx may be too small, downHeap() will fix
    the heap in log(N) time.
    """

    nn = len(heap) - 1
    limit = nn // 2
    val = heap[idx]

    while idx <= limit:
        kid = 2 * idx
        if (kid < nn) and (heap[kid] < heap[kid + 1]):
            kid += 1
        kidVal = heap[kid]
        if kidVal < val:
            break
        heap[idx] = kidVal
        idx = kid
    heap[idx] = val


def heapUpdate(heap, item):
    """
    heapUpdate() does the equivalent of a pop() and push() if the new
    item is less than the max value in the heap.  This is useful when
    using a max-heap to keep track of the N lowest values.
    """

    if item < heap[1]:
        heap[1] = item
        downHeap(heap, 1)


def scramble(val: int) -> int:
    """
    scramble() takes an existing 64-bit hash value and permutes the bits
    into another hash value.  Uses two magic numbers:
    6364136223846793005 = f-value for Mersenne Twister MT19937-64
    9223372036854775783 = largest prime < 2^63
    """

    return ((val * 6364136223846793005) + 9223372036854775783) % 0xffffffffffffffff


def shinglesCalc(text: bytes, window: int = 32, courses: int = 15, tabs: int = 8) -> list[list[int]]:
    """
    shinglesCalc() will process `text` and return a list of variants of
    lists of hashes.  The inner list is often referred to as "shingles"
    and consists for the lowest-value `courses` hashes.  Each top-level
    list represents shingles scrambled `tabs` times.  Conceptually, when
    looking at a section of roof, it's `courses` high and `tabs` wide.
    `window` is the number of bytes in the sliding window that's hashed.
    """

    nn = len(text)
    ww = RkWindow(window)
    seed = ww.filler()
    init = [0xffffffffffffffff] * (courses + 1)
    heaps = [init[:] for _ in range(tabs)]
    for x in text:
        ww.hash(x)
        hh = ww.get()
        for heap in heaps:
            hh = scramble(hh)
            heapUpdate(heap, hh)
    if nn < window:
        for _ in range(window - nn):  # ensure small text gets full singles
            ww.hash(seed)
            hh = ww.get()
            for heap in heaps:
                hh = scramble(hh)
                heapUpdate(heap, hh)
    for heap in heaps:
      heap.pop(0)
      heap.sort()
    return heaps


def vectorCmp(aVec: list[int], bVec: list[int]) -> tuple[int, int]:
    """
    vectorCmp() takes two sorted lists and compares their elements.  The
    tuple returned is (match_count, max_length).
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


def shinglesDist(aa: list[list[int]], bb: list[list[int]]) -> float:
    """
    shinglesDist() is a distance function for two sets of shingles.
    The outputs of shinglesCalc() can be used here.  The return value
    is a real number [0, 1.0] indicating dissimilarity.
    """

    aLen = len(aa)
    bLen = len(bb)
    assert aLen == bLen
    numer = 0
    denom = 0
    for ii in range(aLen):
        n, d = vectorCmp(aa[ii], bb[ii])
        numer += n
        denom += d
    if denom == 0:
        return 1.0
    return (denom - numer) / denom


def simHash(vec: list[int]) -> int:
    """
    simHash() takes one shingle variant (or "tab"), that is, a vector of
    hashes, and returns a single similarity hash as per Moses Charikar's
    2002 paper.  64-bit is assumed.  For proper results, the number of
    elements in the list should be odd, otherwise the bit distribution
    will be skewed.
    """

    nn = len(vec)
    half = nn // 2
    bit = 0x8000000000000000  # 2^63
    rv = 0
    while bit:
        cnt = 0
        for x in vec:
            if x & bit:
                cnt += 1
        if cnt > half:
            rv |= bit
        bit >>= 1
    return rv


def simHashesDist(aa: list[int], bb: list[int]) -> int:
    """
    simHashesDist() compares two lists of SimHashes are returns a distance
    metric.  Each list of SimHashes represents a document.  Corresponding
    elements in each list represent variants or "tabs" of shingles.
    With a SimHash, the most bits in common means the most similar.
    This returns the minimum of the count of differing bits.
    """

    aLen = len(aa)
    bLen = len(bb)
    assert aLen == bLen
    low = 0xffffffffffffffff
    for ii in range(aLen):
        x = aa[ii] ^ bb[ii]
        cnt = x.bit_count()
        if cnt < low:
            low = cnt
    return cnt


def simHashText(text: bytes, window: int = 32, courses: int = 15, tabs: int = 8) -> list[int]:
    """
    Takes text and returns a list of SimHashes.  Arguments:

    text    - The text to process, in UTF-8 bytes
    window  - Width in bytes of the sliding window used for shingles
    courses - The number of least-value shingles to retain
    tabs    - The number of variants of each shingle to process
    """
    assert (courses & 1) == 1
    shingles = shinglesCalc(text, window, courses, tabs)
    return [simHash(hh) for hh in shingles]
