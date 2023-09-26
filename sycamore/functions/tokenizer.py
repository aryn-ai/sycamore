class Tokenizer:
    def tokenize(self, text: str):
        pass


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str):
        return list(text)
