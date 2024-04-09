# De-Duplicating Query Results

The Sycamore near-duplicate detection (NDD) feature can be used to drop duplicates from query results. It is implemented as a [Remote Search Processor](remote_processors.md) called `dedup-response`, and you can add this to your hybrid search or RAG search pipelines. A prerequisite for NDD is to have previously ingested the documents using the `Sketcher` Sycamore transform.  See documentation for [sketch](../data_ingestion_and_preparation/transforms/sketch.md) in `DocSet` for details. Sycamore's defeault data ingestion and search pipeliens have sketching and NDD enabled.

NDD is configured in `pipelines.yml` with a several of preset values:

```yaml
- dedup00:
    processors:
      - dedup-response:
          threshold: 0.01
- dedup01:
    processors:
      - dedup-response:
          threshold: 0.1
- dedup02:
    processors:
      - dedup-response:
          threshold: 0.15
- dedup03:
    processors:
      - dedup-response:
          threshold: 0.2
- dedup04:
    processors:
      - dedup-response:
          threshold: 0.3
- dedup05:
    processors:
      - dedup-response:
          threshold: 0.35
- dedup06:
    processors:
      - dedup-response:
          threshold: 0.4
- dedup07:
    processors:
      - dedup-response:
          threshold: 0.45
- dedup08:
    processors:
      - dedup-response:
          threshold: 0.55
```

As can be seen, there's one parameter, `threshold`, which controls how aggressively NDD will drop documents. Near 0.0, few documents will be removed and they will need to be practically identical to higher-scoring documents. Above 1.0, all documents will be removed, except for the first one.

The current implementation of NDD uses "shingles" which consist of 16 hash values. The distance between two documents is the number of hash values that differ between the two documents' shingles. The raw number is between 0 and 16, but we normalize it to between 0.0 and 1.0. The logic is basically: `if distance < threshold, drop the result`.

Sycamore's default hybrid search and RAG pipelines use `dedup02`, which allows two hashes to differ. That would make the threshold 2 / 16, or 0.125, but we need to set the value slightly higher because it uses a less-than comparison.

The `dedup-response` processor requires the`shingles` field of each retrieved document in the hybrid search step of a pipeline. This will happen by default if the OpenSearch query does not specify `_source`.  Otherwise, it needs to be specified directly:

```
{
  "_source": [
    "shingles",
    "text_representation"
  ],
  "query": {
    "match": {
      "text_representation": "query"
    }
  }
}
```
