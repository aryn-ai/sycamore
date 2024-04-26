#!/bin/bash

echo "Version-Info, Demo UI Branch: ${GIT_BRANCH}"
echo "Version-Info, Demo UI Commit: ${GIT_COMMIT}"
echo "Version-Info, Demo UI Diff: ${GIT_DIFF}"

# run python proxy
if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "ERROR: Missing OPENAI_API_KEY; will not work"
    exit 1
fi

# FIXME: If we unify Dockerfile and git dirs, this will become simpler.
if [[ -f /.dockerenv ]]; then
    PROXYDIR=/home/pn/py-proxy
    UIDIR=/home/pn/js-ui
else
    DIR=$(cd "$(dirname "$0")"; pwd)
    PROXYDIR="${DIR}/openai-proxy"
    UIDIR="${DIR}/ui"
fi

function cleanup {
    kill ${KILLPIDS}
}

function first {
    echo $1
}

cd "${PROXYDIR}"
: ${HOST:=localhost}
SELFSIGNED="${HOST}_ss"
SSLNAME="${SELFSIGNED}"
ARYN_ETC=/etc/opt/aryn

if [[ (! -f ${SELFSIGNED}-key.pem) || (! -f ${SELFSIGNED}-cert.pem) ]]; then
    openssl req -batch -x509 -newkey rsa:4096 -days 10000 \
    -subj "/C=US/ST=California/O=Aryn.ai/CN=${HOST}" \
    -extensions v3_req -addext "subjectAltName=DNS:${HOST}" \
    -noenc -keyout "${SELFSIGNED}-key.pem" \
    -out "${SELFSIGNED}-cert.pem" 2> /dev/null
    echo "Created ${HOST} certificate"
fi

if [[ -f ${ARYN_ETC}/hostcert.pem && -f ${ARYN_ETC}/hostkey.pem ]]; then
    cp -p "${ARYN_ETC}/hostcert.pem" "${HOST}-cert.pem"
    cp -p "${ARYN_ETC}/hostkey.pem" "${HOST}-key.pem"
    SSLNAME="${HOST}"
    echo "Copied certificate from ${ARYN_ETC}"
fi

if [[ ${SSL} == 0 ]]; then
    BASEURL="http://${HOST}:3000"
else
    BASEURL="https://${HOST}:3000"
fi

poetry run python py_proxy/proxy.py "${SSLNAME}" &
KILLPIDS=$!
trap cleanup EXIT

tries=0
while ! curl -k "${BASEURL}/healthz" >/dev/null 2>&1; do
    echo "Demo-UI proxy not running after $tries seconds"
    if [[ $tries -ge 30 ]]; then
        echo "Demo UI probably broken, report a bug on slack"
    fi
    tries=$(expr ${tries} + 1)
    sleep 1
done

# Only in certain environments, run token-protected UI proxy...
if [[ ${SSL} != 0 && -f ${ARYN_ETC}/hostcert.pem ]]; then
    TOKEN=$(openssl rand -hex 24)
    # we should be able to sudo without password here...
    sudo -b poetry run python py_proxy/token_proxy.py "${TOKEN}" "${SSLNAME}"
    # Getting the process ID isn't hard, as the token is new and unique...
    OUT=$(ps -eo pid,cmd | grep "${TOKEN}" | grep -vE 'grep|sudo')
    PID=$(first ${OUT})
    KILLPIDS="${KILLPIDS} ${PID}"
    secs=0
    while ! curl -ks -o /dev/null https://localhost/healthz; do
        echo "Waited ${secs} seconds for token proxy to respond"
        (( ${secs} > 30 )) && echo "Token proxy not working; report on slack"
        sleep 1
        ((++secs))
    done
    URL="https://${HOST}/tok/${TOKEN}"
    echo "${URL}" > token_proxy_url
    (
        for i in {1..10}; do
            echo "token_proxy: ${URL}"
            sleep 5
        done
    ) &
    KILLPIDS="${KILLPIDS} $!"
fi

# run node ui
cd "${UIDIR}"
# Running the UI this way means that the html is unminified and hence easy to read
# Since the UI is open source, there's little downside of doing this and it helps with
# debugging.
# See this link for why we're using RSA self-signed certificates for npm:
# https://community.letsencrypt.org/t/ecdsa-certificates-not-supported-by-create-react-app/201387/9
BROWSER=none PORT=3001 WDS_SOCKET_PORT=3000 HTTPS=true \
SSL_CRT_FILE="${PROXYDIR}/${SELFSIGNED}-cert.pem" \
SSL_KEY_FILE="${PROXYDIR}/${SELFSIGNED}-key.pem" \
npm start | cat

# These are the steps that would build the UI and serve it minified.  We
# should move the build back into the Dockerfile now that we no longer need
# to patch 'localhost' in the source.
#npm run build
#npx serve -n -s build
