# De-Duplicating Query Results

As mentioned in [Remote Search Processors](remote_processors.md), the Sycamore near-duplicate detection (NDD) facility can be used to drop duplicates from query results.  This is implemented as a remote response processor called `dedup-response`.  It's configured in `pipelines.yml` like so:

```yaml
- dedup00:
    processors:
      - dedup-response:
          threshold: 0.0
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

As can be seen, there's one parameter, `threshold`, which is a distance metric that controls how different two documents can be before they're considered not to be duplicates.  A 0.0 means documents must be practically identical.  A 1.0 means all documents are considered to be the same.  Due to limitations in passing parameters through the system, we've pre-created most of the potentially useful setups.

The current implementation of NDD uses "shingles" which consist of 16 hash values.  The `dedup02` preset allows two of those hashes to differ; 0.15 is just a smidge higher than 2 divided by 16.

A prerequisite for query-time NDD is to have previously ingested documents using the `Sketcher` Sycamore transform.  See documentation for [sketch](../data_ingestion_and_preparation/transforms/sketch.md) in DocSet for details.

In order for the `dedup-response` processor to do its job, it must be able to see the `shingles` field of each retrieved document.  This will happen by default if the OpenSearch query does not specify `_source`.  Otherwise, it needs to be listed specifically like so:

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

The practical effect of this is that it's possible to enable or disable NDD by controlling the contents of `_source`.
