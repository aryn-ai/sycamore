#!/bin/bash
set -eu
set -x

groupadd --gid 1000 app
useradd -d /app --uid 1000 --gid app app
chown -R app:app /app
