#!/bin/bash
if [[ ! -f apps/query-ui/queryui/Sycamore_Query.py ]]; then
    echo "Run me as ./apps/query-ui/run-query-ui.sh"
    exit 1
fi
[[ "${QUERY_CACHE}" == "" ]] && export QUERY_CACHE=/app/work/cache_dir
if [[ "${QUERY_CACHE}" == s3://* ]]; then
    # e.g. s3://aryn-temp/llm_cache/luna/ntsb
    :
elif [[ ! -d "${QUERY_CACHE}" ]]; then
    echo "Error: ${QUERY_CACHE} not present."
    echo "Override default cache with QUERY_CACHE environment variable, or if running via docker, either:"
    echo "a) use docker compose up query-ui"
    echo "b) incude:"
    echo '     -mount type=bind,source=\"$(pwd)/apps/query-ui/cache_dir",target=/app/work/cache_dir'
    echo "  in your docker run command line"
    exit 1
fi

if [[ "${OPENSEARCH_HOST}" == "" ]]; then
    if curl -k https://localhost:9200 >/dev/null 2>&1; then
        export OPENSEARCH_HOST=localhost
    elif curl -k https://opensearch:9200 >/dev/null 2>&1; then
        export OPENSEARCH_HOST=opensearch
    else
        echo "Unable to find opensearch at either localhost:9200 or opensearch:9200"
        exit 1
    fi
    echo "Inferred opensearch host ${OPENSEARCH_HOST}"
fi

poetry run python apps/query-ui/external_ray_wrapper.py
