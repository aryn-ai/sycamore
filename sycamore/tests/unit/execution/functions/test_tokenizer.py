import pytest

from sycamore.execution.functions.tokenizer import CharacterTokenizer


class TestTokenOverlapChunker:
    @pytest.mark.parametrize(
        "tokenizer, text, expected_tokens",
        [(CharacterTokenizer(), "a test", ["a", " ", "t", "e", "s", "t"]), (CharacterTokenizer(), "", [])],
    )
    def test_character_tokenizer(self, tokenizer, text, expected_tokens):
        tokens = tokenizer.tokenize(text)
        assert tokens == expected_tokens
