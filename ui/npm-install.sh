#!/bin/bash

npm install
err=$?

echo "DEBUG ${err}"
if [[ ${err} != 0 ]]; then
    ls /root/.npm/_logs
    cat /root/.npm/_logs/*
fi

npm cache clean --force
exit "${err}"
