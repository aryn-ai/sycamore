#!/bin/bash
main() {
    config
    cd notebooks
    if [[ "$1" == --fast ]]; then
        echo "Testing fast notebooks..."
        local start=$(date +%s)
        test_notebooks "${FAST_NOTEBOOKS[@]}"
        local end=$(date +%s)
        local elapsed=$(expr $end - $start)
        echo "-------------------------------------------------------"
        echo "--- Testing fast notebooks took $elapsed/$FAST_MAXTIME seconds"
        if [[ $elapsed -ge $FAST_MAXTIME ]]; then
            echo "--- Fast tests took too long."
            echo "--- Move some fast tests to slow,"
            echo "--- or speed up the fast notebooks."
            exit 1
        fi
        echo "-------------------------------------------------------"
        check_coverage
    elif [[ "$1" == --slow ]]; then
        echo "Testing slow notebooks..."
        test_notebooks "${SLOW_NOTEBOOKS[@]}"
    elif [[ "$1" == --docprep ]]; then
        echo "Testing DocPrep notebooks..."
        test_notebooks "${DOCPREP_NOTEBOOKS[@]}"
    else
        echo "Usage: $0 {--fast,--slow,--docprep}"
        exit 1
    fi
    exit 0
}

config() {
    FAST_NOTEBOOKS=(
        default-prep-script.ipynb # 20+40s on Eric's laptop.
        OpenAI-logprob.ipynb # 5s on Eric's laptop.
    )

    FAST_MAXTIME=120
    SLOW_NOTEBOOKS=(
        ArynPartitionerExample.ipynb # Gets a crash message, but passes?
        ArynPartitionerPython.ipynb
        jupyter_dev_example.ipynb # O(n^2) from re-show/take
        metadata-extraction.ipynb # processes entire 
        sycamore_demo.ipynb # O(n^2) from re-show/take
        tutorial.ipynb # aryn partitioner on entire sort benchmark
        VisualizePartitioner.ipynb
    )
    DOCPREP_NOTEBOOKS=(
        docprep/minilm-l6-v2_greedy-section-merger_duckdb.ipynb
        docprep/minilm-l6-v2_greedy-section-merger_opensearch.ipynb
        docprep/minilm-l6-v2_greedy-text-element-merger_duckdb.ipynb
        docprep/minilm-l6-v2_marked-merger_duckdb.ipynb
        docprep/text-embedding-3-small_greedy-section-merger_duckdb.ipynb
        docprep/text-embedding-3-small_greedy-section-merger_pinecone.ipynb
        docprep/text-embedding-3-small_greedy-text-element-merger_opensearch.ipynb
        docprep/text-embedding-3-small_marked-merger_pinecone.ipynb
    )
    EXCLUDE_NOTEBOOKS=(
        # No good reason for exclusion, just what was
        # not automatically tested before
        ArynPartitionerWithLangchain.ipynb # needs langchain to be installed
        duckdb-writer.ipynb # timeout
        elasticsearch-writer.ipynb # needs elasticsearch db setup
        financial-docs-10k-example.ipynb # needs to be fixed like
        # tutorial to have assertion, assertion code needs to move to aryn_sdk
        ndd_example.ipynb #broken
        ntsb-demo.ipynb # needs rps
        opensearch_docs_etl.ipynb # timeout
        opensearch-writer.ipynb # depends on langchain
        pinecone-writer.ipynb # needs pinccone key
        query-demo.ipynb # depnds on un-checked in visualize.py
        subtask-sample.ipynb # broken -- old style llm prompts
        sycamore-tutorial-intermediate-etl.ipynb # timeout
        unpickle_query.ipynb # looking for uncommitted file
        weaviate-writer.ipynb # path not set to files
    )
}

test_notebooks() {
    for i in "$@"; do
        echo
        echo
        echo "-------------------------------------------------------------------------"
        echo "Starting test on $i as written"
        echo "-------------------------------------------------------------------------"

        time poetry run pytest --nbmake --nbmake-timeout=600 $i || exit 1

        if [[ $(grep -c sycamore.EXEC_LOCAL $i) -ge 1 ]]; then
            sed -e 's/sycamore.EXEC_LOCAL/sycamore.EXEC_RAY/' <$i >ray-variant-$i
            echo "-------------------------------------------------------------------------"
            echo "Starting test on $i with EXEC_RAY"
            echo "-------------------------------------------------------------------------"
            time poetry run pytest --nbmake --nbmake-timeout=600 ray-variant-$i || exit 1
            rm ray-variant-$i
        fi
    done
}

check_coverage() {
    echo "Verifying coverage of all notebooks..."
    find . -name '*.ipynb' | sed 's|^\./||' | grep -v '^ray-variant-' > /tmp/notebooks.list
    (
        for i in "${FAST_NOTEBOOKS[@]}" "${SLOW_NOTEBOOKS[@]}" "${DOCPREP_NOTEBOOKS[@]}" "${EXCLUDE_NOTEBOOKS[@]}"; do
            echo "$i"
        done
    ) >>/tmp/notebooks.list
    sort /tmp/notebooks.list | uniq -c | sort -n | grep -v '^      2 ' >/tmp/notebooks.unlisted
    if [[ $(wc -l </tmp/notebooks.unlisted) != 0 ]]; then
        echo "ERROR: some notebooks are unlisted."
        echo "   Add them to FAST_NOTEBOOKS for fast tests that block commits,"
        echo "   SLOW_NOTEBOOKS for slower tests that do not block commits,"
        echo "   DOCPREP_NOTEBOOKS for DocPrep  tests,"
        echo "   or EXCLUDE_NOTEBOOKS for notebooks that should not be automatically tested."
        echo "   Missing notebooks:"
        sed 's/^      1 //' /tmp/notebooks.unlisted
        exit 1
    fi
    rm /tmp/notebooks.list /tmp/notebooks.unlisted
    echo "All notebooks are classified as fast, slow, docprep, or excluded."
}


main "$@"
exit 1
