# pip install google-cloud-bigquery google-cloud-storage
import argparse
import os
from google.cloud import storage
from google.cloud import bigquery


def get_objects(uri_prefix: str):
    with open("/tmp/hash_name.txt", "w") as f:
        print(f"Getting objects under {uri_prefix}...")
        bucket, prefix = uri_prefix.removeprefix("gs://").split("/", 1)
        storage_client = storage.Client()

        blobs = storage_client.list_blobs(bucket, prefix=prefix)

        skip_count = 0
        objects = {}
        for blob in blobs:
            if "large-results" in blob.name:
                skip_count += 1
                if (skip_count % 1000) == 0:
                    print(f"  {skip_count} skipped large-results")
                continue
            print(f"{blob.md5_hash} {blob.crc32c} {blob.name} {blob.size}", file=f)
            # blob.name gives the full path within the bucket
            objects[f"gs://{bucket}/{blob.name}"] = [blob.md5_hash, blob.size]
            if (len(objects) % 1000) == 0:
                print(f"  {len(objects)} so far...")

        return objects


def get_existing_uris() -> dict[str, tuple[str, int]]:
    print("Getting existing uris...")
    client = bigquery.Client()
    query_job = client.query("SELECT uri, checksum, size FROM example.input_files")

    uris: dict[str, tuple[str, int]] = {}
    for row in query_job:
        assert row.uri and row.checksum and row.size is not None
        uris[row.uri] = (row.checksum, row.size)
        if (len(uris) % 1000) == 0:
            print(f"  {len(uris)} so far...")

    return uris


# If you do this, then trying to manipulate the table gives an error around streaming buffer
def bad_insert_uris(uris):
    if not uris:
        print("No new URIs found")
        return

    print("Inserting uris...")
    bigquery_client = bigquery.Client()
    table_ref = bigquery_client.dataset("example").table("documents")

    rows = []
    for uri in uris:
        rows.append({"uri": uri})

    errors = bigquery_client.insert_rows_json(table_ref, rows)
    assert errors == [], f"Error during insert: {errors}"
    print(f"Added {len(rows)} URIs")


def insert_uris(existing, additional, verbose=False):
    rows = []
    for k in additional:
        if data := existing.get(k):
            assert data == additional[k], f"Mismatch on {k}: {data} vs {additional[k]}"
        else:
            rows.append({"uri": k, "checksum": additional[k][0], "size": additional[k][1]})

    if len(rows) == 0:
        print("No new URIs found")
        return

    print(f"Found {len(rows)} new URIs to add")

    if verbose:
        print("URIs to be added:")
        for row in rows:
            print(f"  {row['uri']}")

    try:
        response = (
            input(f"\nDo you want to add these {len(rows)} URIs (use --verbose to see them)? (yes/no): ")
            .strip()
            .lower()
        )
        if response not in ["yes", "y"]:
            print("Operation cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")
        return

    print("Creating temporary table for uris...")
    bigquery_client = bigquery.Client()

    import uuid

    temp_table_id = f"temp_uris_{uuid.uuid4().hex[:8]}"
    temp_table_ref = bigquery_client.dataset("example").table(temp_table_id)

    schema = [
        bigquery.SchemaField("uri", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("checksum", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("size", "INT64", mode="REQUIRED"),
    ]

    # Create temporary table
    temp_table = bigquery.Table(temp_table_ref, schema=schema)
    temp_table = bigquery_client.create_table(temp_table)

    try:
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        job = bigquery_client.load_table_from_json(rows, temp_table_ref, job_config=job_config)
        job.result()  # Force completion

        print("Inserting temporary table into main table")
        insert_query = f"""
        INSERT INTO example.input_files(uri, checksum, size)
        SELECT uri, checksum, size FROM example.{temp_table_id}
        """

        query_job = bigquery_client.query(insert_query)
        query_job.result()

        print(f"Added {len(rows)} URIs")
    finally:
        print("Cleaning up temporary table")
        bigquery_client.delete_table(temp_table_ref)


def add_missing_objects(uri_prefix, verbose=False):
    objects = get_objects(uri_prefix)
    existing = get_existing_uris()
    insert_uris(existing, objects, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate URIs from GCS prefix into BigQuery")
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="Print URIs that would be added (default: False)"
    )
    parser.add_argument(
        "--uri-prefix",
        default=os.environ.get("ARYN_BQ_INPUT_PREFIX"),
        help="GCS URI prefix to scan (default: ARYN_BQ_INPUT_PREFIX env var)",
    )

    args = parser.parse_args()

    if not args.uri_prefix:
        print("Error: No URI prefix specified. Set ARYN_BQ_INPUT_PREFIX environment variable or use --uri-prefix")
        exit(1)

    add_missing_objects(args.uri_prefix, args.verbose)
