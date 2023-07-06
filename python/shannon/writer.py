from shannon import Context
from shannon.execution import Node


class DocSetWriter:
    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def opensearch(
            self,
            *,
            url: str,
            index: str):
        from shannon.execution.writes import OpenSearchWrite
        os = OpenSearchWrite(self.plan, url=url, index=index)
        os.execute()
