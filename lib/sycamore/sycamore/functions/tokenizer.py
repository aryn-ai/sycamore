from abc import ABC, abstractmethod
from functools import cache
from typing import Union, Optional


class Tokenizer(ABC):
    def __init__(self, max_tokens: Optional[int] = None):
        # TODO: Make max_tokens non-optional
        self.max_tokens = max_tokens

    @cache
    @abstractmethod
    def tokenize(self, text: str, as_ints: bool = False) -> Union[list[int], list[str]]:
        pass


class OpenAITokenizer(Tokenizer):
    def __init__(self, model_name: str, max_tokens: Optional[int] = None, lazy_load: bool = True):
        self._model_name = model_name
        self._tk = None
        if not lazy_load:
            self._load_tk()
        super().__init__(max_tokens)

    def _load_tk(self):
        if self._tk is None:
            import tiktoken

            self._tk = tiktoken.encoding_for_model(self._model_name)

    @cache
    def tokenize(self, text: str, as_ints: bool = False):
        self._load_tk()
        assert self._tk is not None, "Type narrowing, unreachable"
        token_ids = self._tk.encode(text)
        if as_ints:
            return token_ids
        tokens = self._tk.decode_batch([[id] for id in token_ids])
        return tokens


class CharacterTokenizer(Tokenizer):
    @cache
    def tokenize(self, text: str, as_ints: bool = False):
        if as_ints:
            return [ord(c) for c in text]
        return list(text)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self._tk = AutoTokenizer.from_pretrained(model_name)
        super().__init__(max_tokens=self._tk.model_max_length)

    @cache
    def tokenize(self, text: str, as_ints: bool = False):
        if as_ints:
            return self._tk.encode(text)
        return self._tk.tokenize(text)
