# This file contains Sycamore Query UI configuration for use with NTSB Incident Report data.
# Different datasets and applications will need to customize these settings.

from typing import List

from sycamore.query.planner import PlannerExample
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.count import Count
from sycamore.query.operators.top_k import TopK
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaField


# Example queries shown as part of the welcome message.
EXAMPLE_QUERIES = [
    "How many incidents were there in Washington in 2023?",
    "What was the breakdown of aircraft types for incidents with substantial damage?",
    "Show me incidents involving Piper aircraft",
    "Show the details on accident ERA23LA153",
]

# The welcome message shown at the top of a chat session.
WELCOME_MESSAGE = f"""Welcome to the NTSB incident query demo! You can ask me questions about
[NTSB incident reports](https://carol.ntsb.gov/), and I'll do my best to answer them. Feel free
to ask about specific incidents, aggregate statistics, or anything else you're curious about.
If you're not sure what to ask, you can try one of the following example queries:

{"".join([f"<SuggestedQuery query='{query}' />" for query in EXAMPLE_QUERIES])}
"""

# The top-level system prompt provided to the chat agent LLM.
SYSTEM_PROMPT = """You are a helpful agent that answers questions about NTSB
(National Transportation Safety Board) incidents. You have access to a database of incident
reports, each of which has an associated PDF document, as well as metadata about the incident
including the location, date, aircraft type, and more. You can answer questions about the
contents of individual reports, as well as aggregate statistics about the incidents in the
database. You can perform actions such as filtering, sorting, and aggregating the data to
answer questions. You can also provide links to relevant documents and data sources."""

# The index name used for Planner query examples.
PLANNER_EXAMPLE_INDEX = "const_ntsb"

# The example schema used for the Planner query examples.
PLANNER_EXAMPLE_SCHEMA = OpenSearchSchema(
    fields={
        "text_representation": OpenSearchSchemaField(
            field_type="str", description="Can be assumed to have all other details"
        ),
        "properties.entity.dateTime": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "2023-01-12T11:00:00",
                "2023-01-11T18:09:00",
                "2023-01-10T16:43:00",
                "2023-01-28T19:02:00",
                "2023-01-12T13:00:00",
            ],
        ),
        "properties.entity.dateAndTime": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "January 28, 2023 19:02:00",
                "January 10, 2023 16:43:00",
                "January 11, 2023 18:09:00",
                "January 12, 2023 13:00:00",
                "January 12, 2023 11:00:00",
            ],
        ),
        "properties.entity.lowestCeiling": OpenSearchSchemaField(
            field_type="str",
            examples=["Broken 3800 ft AGL", "Broken 6500 ft AGL", "Overcast 500 ft AGL", "Overcast 1800 ft AGL"],
        ),
        "properties.entity.aircraftDamage": OpenSearchSchemaField(
            field_type="str",
            examples=["Substantial", "None", "Destroyed"],
        ),
        "properties.entity.conditions": OpenSearchSchemaField(
            field_type="str",
            examples=["Instrument (IMC)", "IMC", "VMC", "Visual (VMC)"],
        ),
        "properties.entity.departureAirport": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "Somerville, Tennessee",
                "Colorado Springs, Colorado (FLY)",
                "Yelm; Washington",
                "Winchester, Virginia (OKV)",
                "San Diego, California (KMYF)",
            ],
        ),
        "properties.entity.accidentNumber": OpenSearchSchemaField(
            field_type="str",
            examples=["CEN23FA095", "ERA2BLAT1I", "WPR23LA088", "ERA23FA108", "WPR23LA089"],
        ),
        "properties.entity.windSpeed": OpenSearchSchemaField(
            field_type="str",
            examples=["", "10 knots", "7 knots", "knots", "19 knots gusting to 22 knots"],
        ),
        "properties.entity.day": OpenSearchSchemaField(
            field_type="str",
            examples=["2023-01-12", "2023-01-10", "2023-01-20", "2023-01-11", "2023-01-28"],
        ),
        "properties.entity.destinationAirport": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "Somerville, Tennessee",
                "Yelm; Washington",
                "Agua Caliente Springs, California",
                "Liberal, Kansas (LBL)",
                "Alabaster, Alabama (EET)",
            ],
        ),
        "properties.entity.location": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "Hooker, Oklahoma",
                "Somerville, Tennessee",
                "Yelm; Washington",
                "Agua Caliente Springs, California",
                "Dayton, Virginia",
            ],
        ),
        "properties.entity.operator": OpenSearchSchemaField(
            field_type="str",
            examples=["On file", "First Team Pilot Training LLC", "file On", "Anderson Aviation LLC", "Flying W Ranch"],
        ),
        "properties.entity.temperature": OpenSearchSchemaField(
            field_type="str",
            examples=["18'C /-2'C", "15.8C", "13'C", "2C / -3C"],
        ),
        "properties.entity.visibility": OpenSearchSchemaField(
            field_type="str",
            examples=["", "miles", "0.5 miles", "7 miles", "10 miles"],
        ),
        "properties.entity.aircraft": OpenSearchSchemaField(
            field_type="str",
            examples=["Piper PA-32R-301", "Beech 95-C55", "Cessna 172", "Piper PA-28-160", "Cessna 180K"],
        ),
        "properties.entity.conditionOfLight": OpenSearchSchemaField(
            field_type="str",
            examples=["", "Night/dark", "Night", "Day", "Dusk"],
        ),
        "properties.entity.windDirection": OpenSearchSchemaField(
            field_type="str",
            examples=["", "190°", "200", "2005", "040°"],
        ),
        "properties.entity.lowestCloudCondition": OpenSearchSchemaField(
            field_type="str",
            examples=["", "Broken 3800 ft AGL", "Overcast 500 ft AGL", "Clear", "Overcast 200 ft AGL"],
        ),
        "properties.entity.injuries": OpenSearchSchemaField(
            field_type="str",
            examples=["Minor", "Fatal", "None", "3 None", "2 None"],
        ),
        "properties.entity.flightConductedUnder": OpenSearchSchemaField(
            field_type="str",
            examples=[
                "Part 91: General aviation Instructional",
                "Part 135: Air taxi & commuter Non-scheduled",
                "Part 91: General aviation Personal",
                "Part 135: Air taxi & commuter Scheduled",
                "Part 91: General aviation Business",
            ],
        ),
    }
)

# The following to be replaced by config file.
# https://github.com/aryn-ai/sycamore/pull/843
PLANNER_EXAMPLES: List[PlannerExample] = [
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="List the incidents in Georgia in 2023.",
            result_node=0,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports",
                    index=PLANNER_EXAMPLE_INDEX,
                    node_id=0,
                    query={
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "properties.entity.dateTime": {
                                            "gte": "2023-01-01T00:00:00",
                                            "lte": "2023-12-31T23:59:59",
                                            "format": "strict_date_optional_time",
                                        }
                                    }
                                },
                                {"match": {"properties.entity.location": "Georgia"}},
                            ]
                        }
                    },
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="Show the incidents involving Piper aircraft.",
            result_node=0,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports",
                    index=PLANNER_EXAMPLE_INDEX,
                    node_id=0,
                    query={"match": {"properties.entity.aircraft": "Piper"}},
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="How many incidents happened in clear weather?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports in clear weather",
                    index=PLANNER_EXAMPLE_INDEX,
                    node_id=0,
                    query={"match": {"properties.entity.conditions": "VMC"}},
                ),
                1: Count(
                    description="Count the number of incidents",
                    distinct_field="properties.entity.accidentNumber",
                    inputs=[0],
                    node_id=1,
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="What types of aircrafts were involved in accidents in California?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports in California",
                    index=PLANNER_EXAMPLE_INDEX,
                    query={"match": {"properties.entity.location": "California"}},
                    node_id=0,
                ),
                1: TopK(
                    description="Get the types of aircraft involved in incidents in California",
                    field="properties.entity.aircraft",
                    primary_field="properties.entity.accidentNumber",
                    K=100,
                    descending=False,
                    llm_cluster=False,
                    llm_cluster_instruction=None,
                    inputs=[0],
                    node_id=1,
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="Which aircraft accidents in California in 2023 occurred when the wind was stronger than 4 knots?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports in California in 2023",
                    index=PLANNER_EXAMPLE_INDEX,
                    query={
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "properties.entity.dateTime": {
                                            "gte": "2023-01-01T00:00:00",
                                            "lte": "2023-12-31T23:59:59",
                                            "format": "strict_date_optional_time",
                                        }
                                    }
                                },
                                {"match": {"properties.entity.location": "California"}},
                            ]
                        }
                    },
                    node_id=0,
                ),
                1: LlmFilter(
                    description="Filter to reports with wind speed greater than 4 knots",
                    question="Is the wind speed greater than 4 knots?",
                    field="properties.entity.windSpeed",
                    inputs=[0],
                    node_id=1,
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="Which three aircraft types were involved in the most accidents?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports",
                    index=PLANNER_EXAMPLE_INDEX,
                    node_id=0,
                    query={"match_all": {}},
                ),
                1: TopK(
                    description="Get the top three aircraft types involved in accidents",
                    field="properties.entity.aircraft",
                    primary_field="properties.entity.accidentNumber",
                    K=3,
                    descending=True,
                    llm_cluster=False,
                    llm_cluster_instruction=None,
                    inputs=[0],
                    node_id=1,
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="Show some incidents where pilot training was mentioned as a cause",
            result_node=0,
            nodes={
                0: QueryVectorDatabase(
                    description="Get incident reports mentioning pilot training",
                    index=PLANNER_EXAMPLE_INDEX,
                    query_phrase="pilot training",
                    node_id=0,
                ),
            },
        ),
    ),
    PlannerExample(
        schema=PLANNER_EXAMPLE_SCHEMA,
        plan=LogicalPlan(
            query="Show all incidents involving a Cessna 172 aircraft",
            result_node=0,
            nodes={
                0: QueryDatabase(
                    description="Get all the incident reports involving a Cessna 172 aircraft",
                    index=PLANNER_EXAMPLE_INDEX,
                    query={"match": {"properties.entity.aircraft": "Cessna 172"}},
                    node_id=0,
                ),
            },
        ),
    ),
]
