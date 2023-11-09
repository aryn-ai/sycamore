#!/bin/bash
# run python proxy
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: Missing OPENAI_API_KEY; will not work"
    exit 1
fi

cd /home/pn/py-proxy
poetry run python py_proxy/proxy.py &

# run node ui
cd /home/pn/js-ui
find src -type f -print0 | xargs -r0 sed -i "s/localhost/${LOAD_BALANCER:-localhost}/g"
# Running the UI this way means that the html is unminified and hence easy to read
# Since the UI is open source, there's little downside of doing this and it helps with
# debugging.
npm start

# These are the steps that would build the UI and serve it minified.  Most likely
# we would want to move the npm run build into the dockerfile so that it can happen once.
#npm run build
#npx serve -n -s build
