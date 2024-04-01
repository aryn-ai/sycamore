# Remote Search Processors

> 👉 This page assumes you’re familiar with OpenSearch’s concept of a search pipeline and search processor

Sycamore exposes a number of search processors in addition to the ones built in to OpenSearch. We accomplish this with a search processor in OpenSearch called ‘remote-processor’, which makes network calls to a service hosting the sycamore search processors.

These search processors include:

- `debug`: prints the search response to stdout. Useful for debugging.
- `dedup`: works in conjunction with the `Sketcher` ingest transform to deduplicate search results at query-time. See the [sketch](../data_ingestion_and_preparation/transforms/sketch.md) transform for more details.

These processors run in a microservice in the sycamore ecosystem called RPS. The processors running by default are configured in a config file: [remote-processor-service/config/pipelines.yml](https://github.com/aryn-ai/sycamore/blob/main/apps/remote-processor-service/config/pipelines.yml)

```yaml
- debug:
    processors:
      - debug-response:
- dedup:
    processors:
      - dedup-response:
          threshold: 0.4
```

You can add a remote processor to a search pipeline using standard OpenSearch search pipeline creation syntax:

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

### Configuring RPS

You can change the config file for RPS. Simply declare a processor, which processors are part of it, and their individual parameters. So, maybe I want a processor that dedupes and prints the search response before and after so I can compare them manually. I can declare this configuration in a new `my-pipelines.yml`:

```yaml
- debug-dedup:
    processors:
      - debug-response:
      - dedup-response:
          threshold: 0.3
      - debug-response:
```

Docker-cp in the new config file

```bash
docker container stop sycamore-rps-1 # or whatever the rps container is called in your docker engine
docker cp my-pipelines.yml sycamore-rps-1:/aryn/rps/apps/remote-processor-service/config/pipelines.yml
docker container start sycamore-rps-1
```

And then create the search pipeline like so

```lang-http
PUT /_search/pipeline/pipeline-with-dedup-for-debugging
{
  "response_processors": [
    {
      "remote_processor": {
        "endpoint": "https://rps:2796/RemoteProcessorService/ProcessResponse"
        "processor_name": "debug-dedup"
      }
    }
  ]
}
```
