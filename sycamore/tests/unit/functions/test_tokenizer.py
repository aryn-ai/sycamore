import pytest
import stanza

from sycamore.tests.config import TEST_DIR
from sycamore.functions import CharacterTokenizer
from bs4 import BeautifulSoup
from sycamore.functions.tokenizer import StanzaTokenizer

from stanza.server import CoreNLPClient
from stanza.server.client import StartServer

# stanza.install_corenlp()


class TestCharacterTokenizer:
    @pytest.mark.parametrize(
        "tokenizer, text, expected_tokens",
        [(CharacterTokenizer(), "a test", ["a", " ", "t", "e", "s", "t"]), (CharacterTokenizer(), "", [])],
    )
    def test_character_tokenizer(self, tokenizer, text, expected_tokens):
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens


class TestSentenceAwareTokenizer:
    def test_stanza_tokenizer(self):
        base_path = str(TEST_DIR / "resources/data/htmls/")
        html = open(base_path + "/pt_1.html").read()
        soup = BeautifulSoup(html, "html.parser")
        tokenizer = StanzaTokenizer()
        for token in tokenizer.tokenize(soup.get_text()):
            print(token)

        # only for debugging
        # tokenizer.summarize()

    """
    def test_corenlp(self):
        base_path = str(TEST_DIR / "resources/data/htmls/")
        html = open(base_path + "/pt_1.html").read()
        soup = BeautifulSoup(html, "html.parser")
        with CoreNLPClient(
            start_server=StartServer.DONT_START,
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'],
            timeout=30000,
            memory='4G'
        ) as client:
            ann = client.annotate(soup.get_text())
            # print(ann.sentence[0])
            # print(ann.sentence[0].token[0])
            for sentence in ann.sentence:
                for token in sentence.token:
                    print(f'{token.word}\t{token.pos}\t{token.ner}')
    """
