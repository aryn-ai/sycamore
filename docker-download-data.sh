#!/bin/sh
set -eu
set -x

mkdir -p /app/.scrapy/downloads/pdf
cp /app/docker-preload-models.pdf /app/.scrapy/downloads/pdf
#poetry run python examples/docker_local_ingest.py /app/.scrapy
