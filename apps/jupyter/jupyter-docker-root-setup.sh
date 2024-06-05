#!/bin/bash
set -eu
set -x

[[ $(pwd) == /app ]]

if [[ ! -r /app/.git.commit.${GIT_COMMIT} ]]; then
    cat <<EOF
ERROR: Missing /app/.git.commit.${GIT_COMMIT} file

This means you've got an inconsistency between the base sycamore image and the
expected version for this Dockerfile.

Usually this means you need
  --build-arg=TAG=_something_
in your build command

Matching files:
EOF
    ls /app/.git.commit.*
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
sed -i -e 's/^Components: main/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
apt update
apt -y install --no-install-recommends fonts-liberation less sudo groff-base nvidia-cuda-toolkit-gcc libnvidia-tesla-cuda1
