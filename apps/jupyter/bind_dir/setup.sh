#!/bin/bash
echo "Starting setup script."
echo "Usually used to enable optional sycamore features."
set -x
set -e

# Examples of installing additional OS packages:
export DEBIAN_FRONTEND=noninteractive
# sudo apt update && sudo apt -y install --no-install-recommends less

# Examples of setting up optional features:
# ./poetry-install.sh duckdb
# ./poetry-install.sh weaviate

# Setup a mapped in copy for a git checkout of sycamore
maybelink() {
    if [[ -L $2  ]]; then
        :
    else
        if [[ -e $2 ]]; then
            mv $2 $2.orig
            ln -s $1 $2
        fi
    fi
}

if [[ -f /app/sycamore.git/README.md ]]; then
    echo "Found a sycamore git checkout; replacing the preinstalled docker one..."
    for i in README.md apps lib poetry.lock pyproject.toml; do
        maybelink sycamore.git/$i /app/$i
    done
    for i in examples notebooks; do
        maybelink ../sycamore.git/$i /app/work/$i
    done
    # Reinstall everything that's been installed
    for i in $(cat /app/.poetry.install); do
        ./poetry-install.sh $(echo $i | sed 's/,/ /')
    done
fi
