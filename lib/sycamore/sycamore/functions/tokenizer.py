import tiktoken
from transformers import AutoTokenizer


class Tokenizer:
    def tokenize(self, text: str):
        pass


class OpenAITokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self._tk = tiktoken.encoding_for_model(model_name)

    def tokenize(self, text: str):
        return self._tk.encode(text)


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str):
        return list(text)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self._tk = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str):
        return self._tk.tokenize(text)
