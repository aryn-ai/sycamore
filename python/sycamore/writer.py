from typing import Dict

from sycamore import Context
from sycamore.execution import Node


class DocSetWriter:
    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def opensearch(
            self,
            *,
            os_client_args: Dict,
            index_name: str,
            index_settings: Dict = None) -> None:
        from sycamore.execution.writes import OpenSearchWriter
        os = OpenSearchWriter(
            self.plan,
            index_name,
            os_client_args=os_client_args,
            index_settings=index_settings)
        os.execute()
