from sycamore.functions.rabin_karp import (
    RkHash,
    RkWindow,
    shinglesCalc,
    shinglesDist,
    simHashText,
    simHashesDist,
)


class TestRabinKarp:
    texts = [
        "You don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain’t no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly—Tom’s Aunt Polly, she is—and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before.",
        "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly--Tom's Aunt Polly, she is--and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before.",
        "That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly—Tom’s Aunt Polly, she is—and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before. You don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain’t no matter.",
        "Now the way that the book winds up is this: Tom and me found the money that the robbers hid in the cave, and it made us rich. We got six thousand dollars apiece—all gold. It was an awful sight of money when it was piled up. Well, Judge Thatcher he took it and put it out at interest, and it fetched us a dollar a day apiece all the year round—more than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in all her ways; and so when I couldn’t stand it no longer I lit out.",
    ]


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
        bb.hashOut(100)
        bb.hashIn(132)

        assert aa.get() == bb.get()


    def test_window(self):
        aa = RkHash(32)
        for ch in range(101, 133):
            aa.hashIn(ch)

        ww = RkWindow(32)
        for ch in range(100, 133):
            ww.hash(ch)

        assert aa.get() == ww.get()


    def test_similarity(self):
        shingles = []
        simHashes = []
        for text in self.texts:
            utf = text.encode("utf-8")
            shingles.append(shinglesCalc(utf))
            simHashes.append(simHashText(utf))

        # make sure it's symmetric and reflexive
        for ii in range(len(self.texts)):
            for jj in range(ii):
                fwd = shinglesDist(shingles[ii], shingles[jj])
                rev = shinglesDist(shingles[jj], shingles[ii])
                assert fwd == rev
                abc = simHashesDist(simHashes[ii], simHashes[jj])
                cba = simHashesDist(simHashes[jj], simHashes[ii])
                assert abc == cba
            assert shinglesDist(shingles[ii], shingles[ii]) == 0.0
            assert simHashesDist(simHashes[ii], simHashes[ii]) == 0.0

        # make sure examples measure as expected
        assert shinglesDist(shingles[0], shingles[1]) < 0.3
        assert shinglesDist(shingles[0], shingles[2]) < 0.2
        assert shinglesDist(shingles[1], shingles[2]) < 0.4
        assert shinglesDist(shingles[0], shingles[3]) > 0.9
        assert shinglesDist(shingles[1], shingles[3]) > 0.9
        assert shinglesDist(shingles[2], shingles[3]) > 0.9
        assert simHashesDist(simHashes[0], simHashes[1]) < 13
        assert simHashesDist(simHashes[0], simHashes[2]) < 11
        assert simHashesDist(simHashes[1], simHashes[2]) < 14
        assert simHashesDist(simHashes[0], simHashes[3]) > 29
        assert simHashesDist(simHashes[1], simHashes[3]) > 29
        assert simHashesDist(simHashes[2], simHashes[3]) > 29
