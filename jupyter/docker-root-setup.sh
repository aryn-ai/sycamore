#!/bin/bash
set -eu
set -x

groupadd --gid 1000 app
useradd -d /app --uid 1000 --gid app app
chown -R app:app /app

# Once we chown over as part of fixuser, sudo will be unhappy because the user won't exist.
echo 'app1000:x:1000:1000::/app:/bin/sh' >>/etc/passwd
echo 'app1000:!:19697:0:99999:7:::' >>/etc/shadow

export DEBIAN_FRONTEND=noninteractive

proxy=http://172.17.0.1:3128/

http_proxy="${proxy}" curl -o /tmp/proxy http://deb.debian.org/debian || true
ls -l /tmp/proxy || true
if [[ $(grep -c debian.org /tmp/proxy) -ge 1 ]]; then
    export http_proxy="${proxy}"
fi

apt update
apt -y install --no-install-recommends fonts-liberation less sudo

apt clean
rm -rf /var/lib/apt/lists/*
