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
        seed = self.filler()
        ary = []
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
        return 149  # 8-bit prime with 4 bits set
