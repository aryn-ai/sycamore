Scripts for processing a large number of files on Aryn DocParse via BigQuery

* sync_code_with_bigquery.py: Synchronize the local code to the code in bigquery, showing a diff before uploading.
    * queue_async.py: UDF for queuing a new async job
    * get_status.py: UDF for getting status of an async job
    * sleep.py: UDF for sleeping
    * stored_procedures.sql: stored procedures for doing the processing.
* info.sh: Extract information about each of the pdf files.
* populate_uris.py: Scan a Prefix and put the uri, checksum and size into the example.input_files
  table.
* download-finals.py: Download all the files we decided to finally process
