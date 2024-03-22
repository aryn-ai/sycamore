#!/bin/bash
set -eu
set -x

export DEBIAN_FRONTEND=noninteractive

apt update
apt -y install --no-install-recommends poppler-utils python3-poetry gcc python3-dev

groupadd --gid 1000 app
useradd -d /app --uid 1000 --gid app app
chown -R app:app /app
