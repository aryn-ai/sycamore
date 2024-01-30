from abc import abstractmethod
from transformers import AutoTokenizer
import stanza

_nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")


class Tokenizer:
    def tokenize(self, text: str):
        pass


class CharacterTokenizer(Tokenizer):
    def tokenize(self, text: str):
        return list(text)


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self._tk = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str):
        return self._tk.tokenize(text)


class SentenceLevelTokenizer(Tokenizer):
    # @abstractmethod
    def sentences(self):
        pass


class StanzaTokenizer(SentenceLevelTokenizer):
    def __init__(self):  # , text: str):
        # self._nlp = _nlp # stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")
        # self._doc = nlp(text)
        self._summary = {}
        # self._nlp = nlp

    """
    def sentences(self):
        for sentence in self._doc.sentences:
            words = []
            for word in sentence.words:
                # print(f"{word.upos}\t{word.text}")
                if word.upos not in self._summary:
                    self._summary[word.upos] = set()
                self._summary[word.upos].add(word.text)

                #print(type(word.upos))
                if word.upos not in ["PUNCT", "SYM"]:
                    #print("Punctuation found: " + word.text)
                #else:
                    words.append(word.text)
            yield ' '.join(words)
    """

    def tokenize(self, text: str):
        doc = _nlp(text)
        tokens = []
        for sentence in doc.sentences:
            words = []
            for word in sentence.words:
                if word.upos not in ["PUNCT", "SYM"]:
                    words.append(word.text)
            tokens.append(" ".join(words))
        return tokens

    def summarize(self):
        """
        Run this on sample data to understand the make-up (POS, etc) and build
        a strategy on how to filter tokens and words (remove punctuations, symbols, stopwords).

        Returns:

        """
        for k, v in self._summary.items():
            print(k)
            print([val for val in v])
