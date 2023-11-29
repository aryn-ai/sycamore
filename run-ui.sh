#!/bin/bash

echo "Version-Info, Demo UI Branch: ${GIT_BRANCH}"
echo "Version-Info, Demo UI Commit: ${GIT_COMMIT}"
echo "Version-Info, Demo UI Diff: ${GIT_DIFF}"

# run python proxy
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: Missing OPENAI_API_KEY; will not work"
    exit 1
fi

cd /home/pn/py-proxy
poetry run python py_proxy/proxy.py &

# run node ui
cd /home/pn/js-ui
# Running the UI this way means that the html is unminified and hence easy to read
# Since the UI is open source, there's little downside of doing this and it helps with
# debugging.
PORT=3001 BROWSER=none npm start

# These are the steps that would build the UI and serve it minified.  We
# should move the build back into the Dockerfile now that we no longer need
# to patch 'localhost' in the source.
#npm run build
#npx serve -n -s build
