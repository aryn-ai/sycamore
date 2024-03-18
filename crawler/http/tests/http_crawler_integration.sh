#!/bin/bash
set -e

cleanup() {
    echo "Starting cleanup"
    if [[ -z "${WORK_DIR}" ]]; then
        echo "  No WORK_DIR, no files to cleanup"
    else
        echo "  Delete ${WORK_DIR}"
        rm -rf "${WORK_DIR}" || true
    fi
    if [[ -z "${HTTP_PID}" ]]; then
        echo "  No HTTP_PID, missing http server"
    else
        echo "  Kill http server ${HTTP_PID}"
        kill -TERM "${HTTP_PID}" || true
    fi
    echo "Finished cleanup"
}

trap "cleanup" EXIT

main() {
    if curl localhost:13756 >/dev/null 2>&1; then
        echo "ERROR: You already have something running on localhost:13756"
        echo "ERROR: It could be a previous run of this test."
        echo "ERROR: To find leftovers: ps -efl | grep test_http_server"
        ps -efl | grep test_http_server
        echo "ERROR: Clean this up so that curl localhost:13756 returns an error"
        exit 1
    fi
    # run test in the http directory
    cd "$(dirname "$0")"/..
    setup

    test_simple
    test_update
    test_prefix
    check_only_localhost

    # put this after phases to keep the working directory around for debugging
    # pause
    exit 0
}

setup() {
    if [ "$(uname -s)" = "Darwin" ]; then
	WORK_DIR=$(mktemp -d -t crawler_test.XXXXXXXXX)
    else
	WORK_DIR=$(mktemp -d --tmpdir crawler_test.XXXXXXXXX)
    fi
    echo "Using workdir ${WORK_DIR}"
    SRC_DIR="${PWD}"
    cd "${WORK_DIR}"
    pwd
    mkdir http_serve work
    cat >http_serve/robots.txt <<'EOF'
User-agent: *
Allow: /
EOF
    echo 'Go to http://aryn.ai/' >http_serve/example1.txt
    echo '<A HREF="http://aryn.ai/">Visit Aryn</A>' >http_serve/example2.html
    cp "${SRC_DIR}/tests/visit_aryn.pdf" http_serve/example3.pdf
    cp "${SRC_DIR}/tests/visit_aryn.pdf" http_serve/example4.qdf
    python3 "${SRC_DIR}/tests/test_http_server.py" >http_server.log 2>&1 &
    HTTP_PID=$!
}

scrape() {
    scrape_complex "$1" -a dest_dir="${WORK_DIR}/work" -a preset=integration_test
}

scrape_complex() {
    what="$1"
    shift
    local log="${WORK_DIR}/scrape.log.$what"
    echo "Scraping for $what"
    (cd "${SRC_DIR}" && poetry run scrapy crawl sycamore "$@" >"${log}" 2>&1) \
        || die "scrape failed"
}

die() {
    echo "ERROR: $@"
    pause
    exit 1
}

test_simple() {
    scrape simple
    [[ -f work/unknown/http:__localhost:13756_example1.txt ]] || die "missing example1"
    [[ -f work/html/http:__localhost:13756_example2.html ]] || die "missing example2"
    [[ -f work/pdf/http:__localhost:13756_example3.pdf ]] || die "missing example3"
    [[ -f work/pdf/http:__localhost:13756_example4.qdf.pdf ]] || die "missing example3"
}

test_update() {
    # make sure last modified seconds changes
    sleep 1
    echo 'updated' >http_serve/example1.txt
    scrape update
    cmp http_serve/example1.txt work/unknown/http:__localhost:13756_example1.txt \
        || die "failed to update"
}

test_prefix() {
    scrape_complex "prefix" -a dest_dir="${WORK_DIR}/prefix" -a url=http://localhost:13756 -a prefix=http://localhost:13756/example3
    [[ -f prefix/pdf/http:__localhost:13756_example3.pdf ]] || die "missing example3"
    [[ $(find prefix -type f -print | grep -c localhost) == 2 ]] || die "Prefix download did not download exactly 2 files"
}

pause() {
    echo -n "return to continue, ctrl-c to abort: "
    read foo
}

check_only_localhost() {
    # only files matching localhost should have been downloaded even though there is a link to
    # aryn.ai in the html sample
    [[ $(find work -type f -print | grep -c -v localhost) == 0 ]] || die "Found non localhost files"
    # expect to have downloaded exactly 6 files (root, robots.txt, 4 examples)
    [[ $(find work -type f -print | grep -c localhost) == 6 ]] || die "Did not download exactly 5 files"
}

main
