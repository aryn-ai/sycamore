from abc import abstractmethod
from typing import Any, List, TypeVar, Generic

T = TypeVar('T')

class Chunker(Generic[T]):
    @abstractmethod
    def chunk(self, tokens: List[T]) -> List[List[T]]:
        pass

class TextOverlapChunker(Chunker[T]):
    """
    TextOverlapChunker is a class for chunking sequences into smaller segments with controlled overlap.
    
    This class inherits from the Chunker class and divides sequences into chunks of a specified size,
    with configurable overlap between adjacent chunks. The implementation ensures that:
    1. All tokens are included in at least one chunk
    2. No chunk exceeds the specified maximum size
    3. Overlap is consistently maintained between chunks
    4. The last chunk is handled correctly even if smaller than chunk_token_count
    
    Args:
        chunk_token_count: The maximum number of tokens to include in each chunk.
        chunk_overlap_token_count: The number of tokens that should overlap between adjacent chunks.
            Must be less than chunk_token_count.
            
    Raises:
        ValueError: If chunk_overlap_token_count >= chunk_token_count or if either parameter is negative.
        
    Example:
        >>> chunker = TextOverlapChunker(chunk_token_count=5, chunk_overlap_token_count=2)
        >>> tokens = list("ABCDEFGHIJK")
        >>> chunks = chunker.chunk(tokens)
        >>> for chunk in chunks: print(''.join(chunk))
        ABCDE
        DEFGH
        GHIJK
    """
    def __init__(self, chunk_token_count: int = 1000, chunk_overlap_token_count: int = 100) -> None:
        super().__init__()
        if chunk_token_count <= 0:
            raise ValueError("Chunk token count must be positive")
        if chunk_overlap_token_count < 0:
            raise ValueError("Chunk overlap token count must be non-negative")
        if chunk_overlap_token_count >= chunk_token_count:
            raise ValueError("Token overlap count between chunks must be less than chunk token count")
            
        self._chunk_token_count = chunk_token_count
        self._chunk_overlap_token_count = chunk_overlap_token_count
        
    def chunk(self, tokens: List[T]) -> List[List[T]]:
        """
        Divide the input sequence into overlapping chunks.
        
        Args:
            tokens: The input sequence to be chunked.
            
        Returns:
            A list of chunks, where each chunk is a list of tokens.
            
        Note:
            The last chunk may be smaller than chunk_token_count but will maintain
            the specified overlap with the previous chunk if possible.
        """
        if not tokens:
            return []
            
        chunks = []
        stride = self._chunk_token_count - self._chunk_overlap_token_count
        
        for start in range(0, len(tokens), stride):
            # Calculate end index for current chunk
            end = min(start + self._chunk_token_count, len(tokens))
            chunk = tokens[start:end]
            
            # Add chunk if it's the first chunk, maintains minimum size, or is the last piece
            if (start == 0 or 
                len(chunk) >= self._chunk_overlap_token_count or 
                end == len(tokens)):
                chunks.append(chunk)
                
            # If we've processed all tokens, break
            if end == len(tokens):
                break
                
        return chunks
