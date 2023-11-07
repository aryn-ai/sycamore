#!/bin/bash

# run python proxy
cd /home/pn/py-proxy
poetry run python py_proxy/proxy.py &

# run node ui
cd /home/pn/js-ui
find src -type f -print0 | xargs -r0 sed -i "s/localhost/${LOAD_BALANCER:-localhost}/g"
npm run build
npx serve -n -s build
