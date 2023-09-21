from typing import Optional

from sycamore import Context
from sycamore.execution import Node


class DocSetWriter:
    def __init__(self, context: Context, plan: Node):
        self.context = context
        self.plan = plan

    def opensearch(
        self, *, os_client_args: dict, index_name: str, index_settings: Optional[dict] = None, **resource_args
    ) -> None:
        from sycamore.execution.writes import OpenSearchWriter

        os = OpenSearchWriter(
            self.plan, index_name, os_client_args=os_client_args, index_settings=index_settings, **resource_args
        )
        os.execute()
