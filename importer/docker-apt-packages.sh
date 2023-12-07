#!/bin/bash
set -eu
set -x

export DEBIAN_FRONTEND=noninteractive

proxy=http://172.17.0.1:3128/

http_proxy="${proxy}" curl -o /tmp/proxy http://deb.debian.org/debian || true
ls -l /tmp/proxy || true
if [[ $(grep -c debian.org /tmp/proxy) -ge 1 ]]; then
    export http_proxy="${proxy}"
fi

apt update
apt -y install --no-install-recommends poppler-utils python3-poetry gcc python3-dev

apt clean
rm -rf /var/lib/apt/lists/*
