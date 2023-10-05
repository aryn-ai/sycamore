from abc import abstractmethod
from typing import Any


class Chunker:
    @abstractmethod
    def chunk(self, tokens: list[Any]) -> list[Any]:
        pass


class TextOverlapChunker(Chunker):
    """
    TextOverlapChunker is a class for chunking text into smaller segments while allowing for token overlap.

    This class inherits from the Chunker class and is designed to divide long text tokens into chunks, each containing
    a specified number of tokens. It allows for a controlled overlap of tokens between adjacent chunks.

    Args:
        chunk_token_count: The maximum number of tokens to include in each chunk.
        chunk_overlap_token_count: The number of tokens that can overlap between adjacent chunks.
            This value must be less than the `chunk_token_count` to ensure meaningful chunking.

    Example:
         .. code-block:: python

            chunker = TextOverlapChunker(chunk_token_count=1000, chunk_overlap_token_count=100)
            chunks = chunker.chunk(data)
    """

    def __init__(self, chunk_token_count: int = 1000, chunk_overlap_token_count: int = 100) -> None:
        super().__init__()
        if chunk_overlap_token_count >= chunk_token_count:
            raise Exception("Token overlap count between chunks must be lesser than chunk token count")
        self._chunk_token_count = chunk_token_count
        self._chunk_overlap_token_count = chunk_overlap_token_count

    def chunk(self, tokens: list[Any]) -> list[Any]:
        return [
            tokens[a : a + self._chunk_token_count]
            for a in range(0, len(tokens), self._chunk_token_count - self._chunk_overlap_token_count)
        ]
