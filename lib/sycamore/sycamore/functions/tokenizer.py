from abc import ABC, abstractmethod
from typing import Union


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, as_ints: bool = False) -> Union[list[int], list[str]]:
        pass


class OpenAITokenizer(Tokenizer):
    def __init__(self, model_name: str):
        import tiktoken

        self._tk = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str, as_ints: bool = False):
        token_ids = self._tk.encode(text)
        if as_ints:
            return token_ids
        tokens = self._tk.decode_batch([[id] for id in token_ids])
        return tokens


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str, as_ints: bool = False):
        if as_ints:
            return [ord(c) for c in text]
        return list(text)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self._tk = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str, as_ints: bool = False):
        if as_ints:
            return self._tk.encode(text)
        return self._tk.tokenize(text)
