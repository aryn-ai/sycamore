from shannon import (DocSet, Context)
from shannon.execution import Node
from shannon.execution.writes import OpenSearchWriter


class TestDocSetWriter:
    def test_opensearch(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, Node())
        execute = mocker.patch.object(OpenSearchWriter, "execute")
        docset.write.opensearch(os_client_args={}, index_name="index")
        execute.assert_called_once()
