#!/bin/bash
set -e
set -x
if [[ ! -d /app/.venv ]]; then
    echo "Missing /app/.venv; assuming no poetry runs so far"
elif [[ $(grep -c VIRTUAL_ENV=./app/.venv .venv/bin/activate) != 1 ]]; then
    echo "ERROR: Broken venv"
    grep VIRTUAL_ENV .venv/bin/activate
    exit 1
fi

dir="$1"
shift
[[ -d "${dir}" ]]
cd $dir
# use one shared venv
[[ -L .venv ]] || ln -snf /app/.venv .

echo ERIC
ls
export POETRY_CACHE_DIR=/tmp/poetry_cache
[[ -f pyproject.toml ]]
[[ -f poetry.lock ]]

if [[ $# = 0 ]]; then
    echo "Usage: $0 <poetry args>"
    exit 1
fi
poetry "$@"
