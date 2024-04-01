# Remote Search Processors

> 👉 For information on OpenSearch search processors, visit the [opensearch documentation](https://opensearch.org/docs/latest/search-plugins/search-pipelines/index/)

Sycamore exposes a number of search processors in addition to the ones built in to OpenSearch.
We accomplish this with a search processor in OpenSearch called ‘remote-processor’.
This processor makes a network call to the Sycamore service hosting the search processors.


These search processors include:

- `dedup`: works in conjunction with the `Sketcher` ingest transform to deduplicate search results at query-time. See the [sketch](../data_ingestion_and_preparation/transforms/sketch.md) transform for more details.
- `debug`: prints the search response to stdout. Useful for debugging.

The processors running by default are configured in a config file: [remote-processor-service/config/pipelines.yml](https://github.com/aryn-ai/sycamore/blob/main/apps/remote-processor-service/config/pipelines.yml).
The default Sycamore search pipelines use the `dedup02` remote search processor, which removes search results that match with higher-scoring search results in 14/16 shingles.

```yaml
- debug:
    processors:
      - debug-response:
- dedup:
    processors:
      - dedup-response:
          threshold: 0.4
```

You can add a hosted remote processor to a search pipeline using standard OpenSearch search pipeline creation syntax:

```lang-http
PUT /_search/pipeline/pipeline-with-remote-dedup
{
  "response_processors": [
    {
      "remote_processor": {
        "endpoint": "https://rps:2796/RemoteProcessorService/ProcessResponse"
        "processor_name": "dedup"
      }
    }
  ]
}
```

The `processor_name` should point to the top-level name of the processor you want to call.
