#!/usr/bin/bash
# Checkpoint of script to deploy script as a cloud function. Incomplete and not yet working.
main() {
    [[ -d examples/bigquery-docparse/get_status ]] || fail "Run me in root dir"
    [[ $(gcloud config get-value project) == lofty-hall-462819-c6 ]] || fail "Wrong project"
    deploy_get_status
    exit 0
}

fail() {
    echo "ERROR: $@"
    exit 1
}

deploy_get_status() {
   gcloud functions deploy get-status-function \
     --runtime python311 \
     --trigger-http \
     --entry-point get_status_entrypoint \
     --region us-central1 \
     --no-allow-unauthenticated \
     --gen2 \
     --source examples/bigquery-docparse/get_status

}

main
