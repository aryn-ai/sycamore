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

i=0
while [[ "${OPENSEARCH_HOST}" == "" ]]; do
    i=$(expr $i + 1)
    if [[ $i -gt 120 ]]; then
        echo "Waiting 120 seconds to find opensearch. Giving up."
        exit 1
    fi
    echo "Searching for opensearch try $i/120..."
    if curl -k https://localhost:9200 >/dev/null 2>&1; then
        export OPENSEARCH_HOST=localhost
    elif curl -k https://opensearch:9200 >/dev/null 2>&1; then
        export OPENSEARCH_HOST=opensearch
    else
        echo "  unable to find opensearch at either localhost:9200 or opensearch:9200"
    fi
    sleep 1
fi

echo "Inferred opensearch host ${OPENSEARCH_HOST}"
    
# poetry run ray start --head
# poetry run python -m streamlit run apps/query-ui/queryui/Sycamore_Query.py
# might also work.  Then move the restart logic in here.
poetry run python apps/query-ui/external_ray_wrapper.py
