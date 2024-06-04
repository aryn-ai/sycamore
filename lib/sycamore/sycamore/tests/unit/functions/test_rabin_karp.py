from sycamore.functions.rabin_karp import RkHash, RkWindow


class TestRabinKarp:
    def test_smoke(self):
        aa = RkHash(3)
        aa.hashIn(101)
        aa.hashIn(102)
        aa.hashIn(103)

        bb = RkHash(3)
        bb.hashIn(100)
        bb.hashIn(101)
        bb.hashIn(102)
        bb.hashOut(100)
        bb.hashIn(103)

        assert aa.get() == bb.get()

    def test_larger(self):
        aa = RkHash(32)
        for ch in range(101, 133):
            aa.hashIn(ch)

        bb = RkHash(32)
        for ch in range(100, 132):
            bb.hashIn(ch)
        bb.hashOutIn(100, 132)

        assert aa.get() == bb.get()

    def test_window(self):
        aa = RkHash(32)
        for ch in range(101, 133):
            aa.hashIn(ch)

        ww = RkWindow(32)
        for ch in range(100, 133):
            ww.hash(ch)

        assert aa.get() == ww.get()
