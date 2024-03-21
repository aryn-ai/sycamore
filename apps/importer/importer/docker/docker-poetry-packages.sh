#!/bin/bash

cat <<'EOF'

In order to speed up repeated builds while testing, the poetry files are cached locally in a janky
way.  The build prints out the mount points at the start, which will let you find the docker
directory under /var/lib/docker. You put a pause file in the virtual tmp directory which causes the
build to pause after the poetry install runs. Then you tar up the cache directory and store it in a
local http server. When the build runs again, the cached file is first downloaded and unpacked which
means that almost all of the files are just accessed locally.

EOF

set -e
set -x
export POETRY_CACHE_DIR=/tmp/poetry.cache
mkdir "${POETRY_CACHE_DIR}"
(
    mount
    cd /tmp
    url=http://172.18.0.1/cache/`uname -m`/sycamore.poetry.tar
    curl -O "${url}" || true
    if [[ $(find sycamore.poetry.tar -size +1000000c -print | grep -c sycamore.poetry.tar) = 1 ]]; then
        # assume if less than 1MB its an error file
        tar xf sycamore.poetry.tar
        du -s "${POETRY_CACHE_DIR}"
    else
        echo "Warning: Missing ${url}, will re-download all files"
        touch sycamore.poetry.tar
    fi
)
find "${POETRY_CACHE_DIR}" >/tmp/poetry.cache.before
poetry install --only main,sycamore_library,docker --no-root -v
find "${POETRY_CACHE_DIR}" >/tmp/poetry.cache.after

diff -u /tmp/poetry.cache.before /tmp/poetry.cache.after >/tmp/poetry.cache.diff || true
head -100 /tmp/poetry.cache.diff
echo "Poetry cache diff lines: $(wc -l </tmp/poetry.cache.diff)"

while [[ -f /tmp/pause ]]; do
    echo "Waiting for /tmp/pause to vanish"
    sleep 5
done

rm -rf "${POETRY_CACHE_DIR}" /tmp/*
