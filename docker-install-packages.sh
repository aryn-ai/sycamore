#!/bin/sh
set -eu
set -x

export DEBIAN_FRONTEND=noninteractive

apt update
apt -y install --no-install-recommends poppler-utils
apt clean
rm -rf /var/lib/apt/lists/*
