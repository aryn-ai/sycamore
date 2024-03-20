#!/bin/bash

apt -y install --no-install-recommends patch
apt clean
rm -rf /var/lib/apt/lists/*

npm install --loglevel verbose
err=$?

if [[ ${err} != 0 ]]; then
    ls /root/.npm/_logs
    cat /root/.npm/_logs/*
    exit 1
fi

# Cleanup files we don't want to keep around. We want to preserve the downloaded file cache

# Don't use npm cache clean --force here because the whole point of having the docker cache is to
# not re-download the files every time and cleaning the cache would defeat that.
rm -rf /root/.npm/_logs /root/.npm/_update-notifier-last-checked

# TODO: https://github.com/aryn-ai/demo-ui/issues/9 - figure out proper fix for worker was terminated.
patch -p0 <pdf.worker.js.patch || exit 1
# map file is wrong once we patch and very confusing in the browser debuggers which shows the
# map file not the actual code.
rm node_modules/pdfjs-dist/build/pdf.worker.js.map
