from shannon.docset import DocSet
import shannon


class TestDocSetReader:

    def test_pdf(self):
        context = shannon.init()
        docset = context.read.binary(
            "s3://bucket/prefix/pdf", binary_format="pdf")
        assert (isinstance(docset, DocSet))
        assert (docset.plan.format() == "pdf")
