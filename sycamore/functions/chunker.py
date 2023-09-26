from abc import abstractmethod
from typing import Any


class Chunker:
    @abstractmethod
    def chunk(self, tokens: list[Any]) -> list[Any]:
        pass


class TokenOverlapChunker(Chunker):
    def __init__(self, chunk_token_count=1000, chunk_overlap_token_count=100) -> None:
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
