from sycamore.query.client import SycamoreQueryClient
from sycamore.query.visualize import visualize_plan
from rich.console import Console

console = Console()
client = SycamoreQueryClient()


OS_INDEX = "const_ntsb"
QUERY = "How many airplane incidents were there in Washington in 2023?"

schema = client.get_opensearch_schema(OS_INDEX)
# console.print(schema)

plan = client.generate_plan(QUERY, OS_INDEX, schema)
# plan.show(verbose=True)
# visualize_plan(plan)

query_id, result = client.run_plan(plan)
console.rule("Query result")
console.print(result)
