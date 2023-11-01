#!/bin/bash
set -e

cleanup() {
    echo "Cleanup ${WORK_DIR} and ${HTTP_PID}..."
    rm -rf "${WORK_DIR}" || true
    [[ -z "${HTTP_PID}" ]] || kill -TERM "${HTTP_PID}" || true
    echo "Clean"
}

trap "cleanup" EXIT

main() {
    # run test in the http directory
    cd "$(dirname "$0")"/..
    setup

    test_simple
    test_update
    check_only_localhost

    # put this after phases to keep the working directory around for debugging
    # pause
    exit 0
}

setup() {
    WORK_DIR=$(mktemp -d --tmpdir crawler_test.XXXXXXXXX)
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
    python3 -m http.server -d http_serve -b localhost 13756 >http_server.log 2>&1 &
    HTTP_PID=$!
}

scrape() {
    local log="${WORK_DIR}/scrape.log.$1"
    echo "Scraping for $1"
    (cd "${SRC_DIR}" && poetry run scrapy crawl aryn -a dest_dir="${WORK_DIR}/work" \
                               -a preset=integration_test >"${log}" 2>&1) \
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
}

test_update() {
    # make sure last modified seconds changes
    sleep 1
    echo 'updated' >http_serve/example1.txt
    scrape update
    cmp http_serve/example1.txt work/unknown/http:__localhost:13756_example1.txt \
        || die "failed to update"
}

pause() {
    echo -n "return to continue, ctrl-c to abort: "
    read foo
}

check_only_localhost() {
    # only files matching localhost should have been downloaded even though there is a link to
    # aryn.ai in the html sample
    [[ $(find work -type f -print | grep -v localhost | wc -l) == 0 ]] || die "Found non localhost files"
    # expect to have downloaded exactly 5 files (root, robots.txt, 3 examples)
    [[ $(find work -type f -print | grep localhost | wc -l) == 5 ]] || die "Found non localhost files"
}

main
