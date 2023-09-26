from sycamore import DocSet, Context
from sycamore.plan_nodes import Node
from sycamore.writes import OpenSearchWriter


class TestDocSetWriter:
    def test_opensearch(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, mocker.Mock(spec=Node))
        execute = mocker.patch.object(OpenSearchWriter, "execute")
        docset.write.opensearch(os_client_args={}, index_name="index")
        execute.assert_called_once()
