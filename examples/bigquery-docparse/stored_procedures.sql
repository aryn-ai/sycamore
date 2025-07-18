# Run udf-submit & udf-status to make the UDFs

CREATE OR REPLACE PROCEDURE example.queue_new_documents()
BEGIN
    UPDATE `example.documents`
    SET async_id = IF(async_id is NULL, example.queue_async(uri, async_id, "0"), async_id),
        tries = IF(async_id is NULL, if(tries is null,0,tries)+1, tries)
    WHERE uri IN (
        # the LIMIT N defines the max in-flight. setting it to a parameter gives a compilation error
        # Gemini's suggested fix is to compute a string as SQL and execute it.
        SELECT uri
        FROM example.documents
        WHERE result is null
          AND (err is null or err not like 'Result too large%')
          AND (uri like '%.pdf' or uri like '%.PDF')
          AND (tries is null or tries < 15) # keep in sync with constant in processing_status
        ORDER BY async_id DESC, processing_order, tries
        LIMIT 50
    );
END;

CREATE OR REPLACE PROCEDURE example.finish_in_flight_documents()
BEGIN
    CREATE OR REPLACE TEMPORARY TABLE in_flight_documents AS
    SELECT (example.get_status(d.async_id)).*  # expand struct into columns
    FROM example.documents as d
    WHERE d.async_id is not null and d.result is null
    and (err is null or err not like 'Result too large%')
 #   and d.async_id != 'aryn:t-q5xs3wcy9bknxy5bez1sqrz' # 250MB
 #   and d.async_id != 'aryn:t-i0uc5w385wlmtx2zuv3uxgn' # 136MB
 #   and d.async_id != 'aryn:t-83ers3f3vxam1g4wz7oqu4o' # 106MB
 #   and d.async_id != 'aryn:t-4wca94rar7j4pyi9loqqmsw' # 98MB
 #   and d.async_id != 'aryn:t-u356i13sinvt6hipukbasgn' # 32MB
 #   and d.async_id != 'aryn:t-8plbhj5vhfu291gg5qf5x8x' # 26MB
 #   and d.async_id != 'aryn:t-nahsaweb6oyz8ekg3ycafo5' # 12MB
    ORDER by rand() LIMIT 1000
    ;

 #   SELECT * from in_flight_documents;
    SELECT async_id, length(result) as result_len, err from in_flight_documents order by err desc, result_len desc;
    UPDATE example.documents as target
    SET async_id = ifd.new_async_id,
        result = ifd.result,
        err = IF(ifd.err is not NULL, ifd.err, target.err)
    FROM in_flight_documents as ifd
    WHERE target.async_id = ifd.async_id;
END;

CREATE OR REPLACE PROCEDURE example.processing_status(OUT remaining_documents INT64)
BEGIN
  DECLARE finished INT64;
  DECLARE remaining INT64;
  DECLARE in_flight INT64;
  DECLARE large INT64;
  DECLARE errs INT64;
  DECLARE try_limit INT64;
  DECLARE status STRING;
  DECLARE sql_query STRING;

  SET (finished, remaining, in_flight, large, try_limit, errs) = (
    SELECT AS STRUCT
      COUNTIF(result is not NULL) as finished,
      COUNTIF(result IS NULL) as remaining,
      COUNTIF(result IS NULL AND async_id is NOT NULL AND (err is null or err not like 'Result too large%')) as in_flight,
      COUNTIF(result like 'gs://%') as large,
      COUNTIF(tries >= 15 and async_id is NULL) as try_limit,
      COUNTIF(err is not null) as errs
    FROM `example.documents`
  );

  SET remaining_documents = remaining - try_limit;
  # include timestamp at the end, but it shows up in End time
  SET status =
   FORMAT("%d in-flight, %d remain, %d finished, %d large, %d try_limit, %d errs at %s",
          in_flight, remaining_documents, finished, large, try_limit, errs,
          FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", CURRENT_TIMESTAMP()));

  SET sql_query = FORMAT('SELECT "%s" AS status_message', status);
  EXECUTE IMMEDIATE sql_query;
END;

CREATE OR REPLACE PROCEDURE example.wait_between_cycles(cycle_start INT64, remaining_count INT64, OUT should_continue BOOL)
BEGIN
  DECLARE exit_below INT64 DEFAULT 0;
  IF remaining_count > exit_below THEN
    SELECT example.sleep_until(cycle_start + 600);
    SET should_continue = TRUE;
  ELSE
    SET should_continue = FALSE;
  END IF;
END;

CREATE OR REPLACE PROCEDURE example.processing_loop()
BEGIN
  DECLARE should_continue BOOLEAN DEFAULT TRUE;
  DECLARE remaining_count INT64 DEFAULT 0;
  DECLARE cycle_start INT64;

  WHILE should_continue DO
    SET cycle_start = UNIX_SECONDS(CURRENT_TIMESTAMP());
    CALL example.queue_new_documents();
    CALL example.finish_in_flight_documents();
    CALL example.processing_status(remaining_count);
    CALL example.wait_between_cycles(cycle_start, remaining_count, should_continue);
  END WHILE;
END;

# CALL example.add_documents_from_gcs("gs://eric-aryn-test-bucket/ntsb/40");
# CALL example.queue_new_documents();
