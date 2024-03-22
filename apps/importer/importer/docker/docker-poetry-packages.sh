#!/bin/bash

set -e
set -x
[[ -f pyproject.toml ]]
[[ -f poetry.lock ]]
if [[ ! -d /app/.venv ]]; then
    echo "Missing /app/.venv; assuming no poetry runs so far"
elif [[ $(grep -c VIRTUAL_ENV=./app/.venv .venv/bin/activate) != 1 ]]; then
    echo "ERROR: Broken venv"
    grep VIRTUAL_ENV .venv/bin/activate
    exit 1
fi
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
