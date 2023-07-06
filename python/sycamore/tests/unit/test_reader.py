from sycamore.docset import DocSet
import sycamore


class TestDocSetReader:

    def test_pdf(self):
        context = sycamore.init()
        docset = context.read.binary(
            "s3://bucket/prefix/pdf", binary_format="pdf")
        assert (isinstance(docset, DocSet))
        assert (docset.plan.format() == "pdf")
