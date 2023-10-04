import pytest

from sycamore.functions import TextOverlapChunker


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
