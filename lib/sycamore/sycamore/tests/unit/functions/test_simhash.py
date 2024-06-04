import sycamore.functions.simhash as sh


class TestShingles:
    texts = [
        "You don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain’t no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly—Tom’s Aunt Polly, she is—and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before.",  # noqa
        "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter. That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly--Tom's Aunt Polly, she is--and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before.",  # noqa
        "That book was made by Mr. Mark Twain, and he told the truth, mainly. There was things which he stretched, but mainly he told the truth. That is nothing. I never seen anybody but lied one time or another, without it was Aunt Polly, or the widow, or maybe Mary. Aunt Polly—Tom’s Aunt Polly, she is—and Mary, and the Widow Douglas is all told about in that book, which is mostly a true book, with some stretchers, as I said before. You don’t know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain’t no matter.",  # noqa
        "Now the way that the book winds up is this: Tom and me found the money that the robbers hid in the cave, and it made us rich. We got six thousand dollars apiece—all gold. It was an awful sight of money when it was piled up. Well, Judge Thatcher he took it and put it out at interest, and it fetched us a dollar a day apiece all the year round—more than a body could tell what to do with. The Widow Douglas she took me for her son, and allowed she would sivilize me; but it was rough living in the house all the time, considering how dismal regular and decent the widow was in all her ways; and so when I couldn’t stand it no longer I lit out.",  # noqa
    ]

    def test_order(self):
        s = "The quick brown fox jumps over the lazy dog"
        shingles = sh.shinglesCalc(s.encode("utf-8"))
        for ii in range(1, len(shingles)):
            assert shingles[ii - 1] <= shingles[ii]

    def test_nonzero(self):
        s = "The quick brown fox jumps over the lazy dog"
        shingles = sh.shinglesCalc(s.encode("utf-8"))
        assert min(shingles) > 0  # zeros are almost always a bug

    def test_shingle_dist(self):
        aa = [1, 2, 3, 4]
        bb = [1, 2, 4, 5]
        cc = [5, 6, 7, 8]
        assert sh.shinglesDist(aa, aa) == 0.0
        assert sh.shinglesDist(aa, bb) == 0.25
        assert sh.shinglesDist(aa, cc) == 1.0

    def test_shingles(self):
        shingles = []
        for text in self.texts:
            utf = text.encode("utf-8")
            shingles.append(sh.shinglesCalc(utf))

        # make sure it's symmetric and reflexive
        for ii in range(len(self.texts)):
            for jj in range(ii):
                fwd = sh.shinglesDist(shingles[ii], shingles[jj])
                rev = sh.shinglesDist(shingles[jj], shingles[ii])
                assert fwd == rev
            assert sh.shinglesDist(shingles[ii], shingles[ii]) == 0.0

        # make sure examples measure as expected
        assert sh.shinglesDist(shingles[0], shingles[1]) < 0.3
        assert sh.shinglesDist(shingles[0], shingles[2]) < 0.3
        assert sh.shinglesDist(shingles[1], shingles[2]) < 0.4
        assert sh.shinglesDist(shingles[0], shingles[3]) > 0.9
        assert sh.shinglesDist(shingles[1], shingles[3]) > 0.9
        assert sh.shinglesDist(shingles[2], shingles[3]) > 0.9

    def test_sim(self):
        shingle = list(range(0x07, 0x16))
        sim = sh.simHash(shingle)
        assert sim == 0x09

    def test_sim_dist(self):
        aa = [0x1111, 0x2222, 0x4444, 0xFFFF]
        bb = [0x1110, 0x2202, 0x4044, 0x0000]
        cc = [0x1100, 0x2002, 0x0044, 0x0000]
        dd = [0x1000, 0x0002, 0x0040, 0x0000]
        ee = [0x0000, 0x0000, 0x0000, 0x0000]
        assert sh.simHashesDist(aa, aa) == 0
        assert sh.simHashesDist(aa, bb) == 19 / 256
        assert sh.simHashesDist(aa, cc) == 22 / 256
        assert sh.simHashesDist(aa, dd) == 25 / 256
        assert sh.simHashesDist(aa, ee) == 28 / 256

    def test_simhashes(self):
        simHashes = []
        for text in self.texts:
            utf = text.encode("utf-8")
            simHashes.append(sh.simHashText(utf))

        # make sure it's symmetric and reflexive
        for ii in range(len(self.texts)):
            for jj in range(ii):
                fwd = sh.simHashesDist(simHashes[ii], simHashes[jj])
                rev = sh.simHashesDist(simHashes[jj], simHashes[ii])
                assert fwd == rev
            assert sh.simHashesDist(simHashes[ii], simHashes[ii]) == 0.0

        # make sure examples measure as expected
        assert sh.simHashesDist(simHashes[0], simHashes[1]) < 0.2
        assert sh.simHashesDist(simHashes[0], simHashes[2]) < 0.2
        assert sh.simHashesDist(simHashes[1], simHashes[2]) < 0.2
        assert sh.simHashesDist(simHashes[0], simHashes[3]) > 0.5
        assert sh.simHashesDist(simHashes[1], simHashes[3]) > 0.5
        assert sh.simHashesDist(simHashes[2], simHashes[3]) > 0.5
