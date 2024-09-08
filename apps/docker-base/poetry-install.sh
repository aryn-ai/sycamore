#!/bin/bash
set -e
set -x
if [[ ! -d /app/.venv ]]; then
    echo "Missing /app/.venv; assuming about to do noop-poetry install"
elif [[ $(grep -c VIRTUAL_ENV=./app/.venv .venv/bin/activate) != 1 ]]; then
    echo "ERROR: Broken venv"
    grep VIRTUAL_ENV .venv/bin/activate
    exit 1
fi

cd /app

if [[ -d /app/work/bind_dir ]]; then
    # Running in a container. Cache to the bind dir for maximum preservation
    export POETRY_CACHE_DIR=/app/work/bind_dir/poetry_cache
else
    # Running during a build. Use the build poetry cache.
    export POETRY_CACHE_DIR=/tmp/poetry_cache
fi

if [[ $# = 0 ]]; then
    echo "Usage: $0 <poetry install packages>"
    exit 1
fi
for i in "$@"; do
    if [[ "$i" == main ]]; then
        :
    elif [[ $(fgrep "[tool.poetry.group.${i}.dependencies]" pyproject.toml | wc -l) != 1 ]]; then
        echo "Unable to find dependencies for $i in pyproject.toml"
        exit 1
    fi
done
only=$(echo "$@" | sed 's/ /,/g')
echo  "$only" >>/app/.poetry.install
poetry install --only $only --no-root -v

