import pytest

from sycamore.functions import TextOverlapChunker
from sycamore.tests.config import TEST_DIR
from bs4 import BeautifulSoup
from sycamore.functions.tokenizer import StanzaTokenizer
from sycamore.functions.chunker import SentenceAwareChunker


class TestTokenOverlapChunker:
    @pytest.mark.parametrize(
        "chunker, tokens, expected_chunks",
        [
            (
                TextOverlapChunker(chunk_token_count=2, chunk_overlap_token_count=1),
                ["a", "b", "c", "d", "e"],
                [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["e"]],
            ),
            (
                TextOverlapChunker(chunk_token_count=2, chunk_overlap_token_count=0),
                ["a", "b", "c", "d", "e"],
                [["a", "b"], ["c", "d"], ["e"]],
            ),
        ],
    )
    def test_token_overlap_chunker(self, chunker, tokens, expected_chunks):
        chunks = chunker.chunk(tokens)
        assert chunks == expected_chunks

    def test_token_overlap_is_greater_than_chunk_size(self):
        with pytest.raises(Exception) as exception:
            TextOverlapChunker(chunk_token_count=2, chunk_overlap_token_count=2)
        assert str(exception.value) == "Token overlap count between chunks must be lesser than chunk token count"

        with pytest.raises(Exception) as exception:
            TextOverlapChunker(chunk_token_count=2, chunk_overlap_token_count=3)
        assert str(exception.value) == "Token overlap count between chunks must be lesser than chunk token count"


class TestSentenceAwareChunker:
    def test_sentence_aware_chunker(self):
        base_path = str(TEST_DIR / "resources/data/htmls/")
        html = open(base_path + "/pt_1.html").read()
        soup = BeautifulSoup(html, "html.parser")
        tokenizer = StanzaTokenizer()
        chunker = SentenceAwareChunker()
        # for i, chunk in enumerate(chunker.get_chunks()):
        #    print(f'Chunk {i}({len(chunk)}):\t{chunk}')

        for i, chunk in enumerate(chunker.chunk(tokenizer.tokenize(soup.get_text()))):
            print(f"Chunk {i}({len(chunk)}):\t{chunk}")
