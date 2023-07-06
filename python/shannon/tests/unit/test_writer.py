from shannon import (DocSet, Context)
from shannon.execution import Node
from shannon.execution.writes import OpenSearchWrite


class TestDocSetWriter:
    def test_opensearch(self, mocker):
        context = mocker.Mock(spec=Context)
        docset = DocSet(context, Node())
        compute = mocker.patch.object(OpenSearchWrite, "execute")
        docset.write.opensearch(url="opensearch", index="index")
        compute.assert_called_once()
