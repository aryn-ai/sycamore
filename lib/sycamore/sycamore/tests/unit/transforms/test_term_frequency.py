from sycamore.data.document import Document
from sycamore.functions.tokenizer import CharacterTokenizer, HuggingFaceTokenizer, OpenAITokenizer
from sycamore.transforms.term_frequency import TermFrequency


class TestTermFrequency:
    def test_tf_with_char_tokenizer(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = CharacterTokenizer()
        tf = TermFrequency(None, tokenizer=tokenizer)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {"a": 1, "e": 2, "l": 1, "m": 1, "p": 1, "s": 1, "t": 2, "x": 1, " ": 1}

    def test_tf_with_char_tokenizer_with_token_ids(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = CharacterTokenizer()
        tf = TermFrequency(None, tokenizer=tokenizer, with_token_ids=True)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {97: 1, 101: 2, 108: 1, 109: 1, 112: 1, 115: 1, 116: 2, 120: 1, 32: 1}

    def test_tf_with_bert_tokenizer(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = HuggingFaceTokenizer(model_name="bert-base-uncased")
        tf = TermFrequency(None, tokenizer=tokenizer)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {"sample": 1, "text": 1}

    def test_tf_with_bert_tokenizer_with_token_ids(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = HuggingFaceTokenizer(model_name="bert-base-uncased")
        tf = TermFrequency(None, tokenizer=tokenizer, with_token_ids=True)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {101: 1, 102: 1, 7099: 1, 3793: 1}

    def test_tf_with_openai_tokenizer(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = OpenAITokenizer(model_name="gpt-3.5-turbo")
        tf = TermFrequency(None, tokenizer=tokenizer)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {"sample": 1, " text": 1}

    def test_tf_with_openai_tokenizer_with_token_ids(self):
        doc = Document()
        doc.text_representation = "sample text"
        tokenizer = OpenAITokenizer(model_name="gpt-3.5-turbo")
        tf = TermFrequency(None, tokenizer=tokenizer, with_token_ids=True)
        tabledoc = tf.run(doc)
        table = tabledoc.properties["term_frequency"]
        assert table == {13925: 1, 1495: 1}
