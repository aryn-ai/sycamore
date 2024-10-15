from sycamore.query.client import SycamoreQueryClient
from rich.console import Console

console = Console()
client = SycamoreQueryClient()


OS_INDEX = "const_ntsb"
QUERY = "How many airplane incidents were there in Washington in 2023?"

schema = client.get_opensearch_schema(OS_INDEX)
# console.print(schema)

plan = client.generate_plan(QUERY, OS_INDEX, schema)
# from sycamore.query.visualize import visualize_plan
# visualize_plan(plan)

# WARNING: As of 2024-09-03, the results are inconsistent; you can get different results
# because of differences in the generated query plan and as a result of differences in the
# processing of the pipeline.
query_id, result = client.run_plan(plan)
console.rule("Query result")
console.print(result)
