#!/bin/bash

set -e
set -x
[[ -f pyproject.toml ]]
[[ -f poetry.lock ]]
export POETRY_CACHE_DIR=/tmp/poetry_cache
# Edit the pyproject.toml so that the poetry setup is under /app. This lets us re-use the
# sycamore installation in the jupyter notebook with reduced confusion on the paths in that
# container.
sed -i 's,../../lib,lib,' pyproject.toml poetry.lock
if [[ $# = 0 ]]; then
    echo "Usage: $0 <poetry args>"
    exit 1
fi
poetry "$@"
