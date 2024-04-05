import logging

# Credit: Claude
# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)

# Create a file handler to write logs to a file
file_handler = logging.FileHandler("test-output.log", mode="w")
file_handler.setLevel(logging.DEBUG)  # Set the file handler logging level

# Create a formatter for log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


def test_querying(query_generator, os_query, opensearch_client, ingested_index):
    logger.info(os_query)
    pipeline, query = query_generator.generate(os_query)
    pipeline_name = "it-pipeline"
    logger.info(pipeline)
    opensearch_client.transport.perform_request(method="PUT", url=f"/_search/pipeline/{pipeline_name}", body=pipeline)
    logger.info(query)
    search_response = opensearch_client.search(
        index=ingested_index.name,
        params={"search_pipeline": pipeline_name, "_source_excludes": "embedding"},
        body=query,
    )
    logger.info(search_response)
