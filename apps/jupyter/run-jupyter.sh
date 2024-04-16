#!/bin/bash

die() {
    echo "ERROR:" "$@" >&2
    exit 1
}

mkdir -p $HOME/.jupyter
if [[ -f /.dockerenv ]]; then
    APP_DIR=/app
else
    APP_DIR=$(cd "$(dirname "$0")"; pwd)
fi
WORK_DIR="${APP_DIR}/work"
BIND_DIR="${WORK_DIR}/bind_dir"
VOLUME="${WORK_DIR}/docker_volume"
RUNTIME_DIR="${HOME}/.local/share/jupyter/runtime"
ARYN_ETC=/etc/opt/aryn
JUPYTER_CONFIG_DOCKER="${VOLUME}/jupyter_notebook_config.py"
mkdir -p "${BIND_DIR}" "${VOLUME}"

if [[ ! -f "${JUPYTER_CONFIG_DOCKER}" ]]; then
    TOKEN=$(openssl rand -hex 24)
    cat >"${JUPYTER_CONFIG_DOCKER}" <<EOF
# Configuration file for notebook.

c = get_config()  #noqa

c.IdentityProvider.token = '${TOKEN}'
EOF
fi
ln -snf "${JUPYTER_CONFIG_DOCKER}" $HOME/.jupyter

find "${RUNTIME_DIR}" -type f -name 'jpserver-*-open.html' -delete 2>/dev/null

if [[ ${SSL} == 0 ]]; then
    echo "Jupyter not serving over SSL."
    SSLARG=()
elif [[ -f ${ARYN_ETC}/hostcert.pem && -f ${ARYN_ETC}/hostkey.pem ]]; then
    echo "Using SSL certificate from ${ARYN_ETC}"
    SSLARG=("--certfile=\"${ARYN_ETC}/hostcert.pem\"" "--keyfile=\"${ARYN_ETC}/hostkey.pem\"")
else
    : ${HOST:=localhost}
    SSLPFX="${VOLUME}/${HOST}"
    SSLLOG="${VOLUME}/openssl.err"
    if [[ (! -f ${SSLPFX}-key.pem) || (! -f ${SSLPFX}-cert.pem) ]]; then
        openssl req -batch -x509 -newkey rsa:4096 -days 10000 \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=${HOST}" \
        -extensions v3_req -addext "subjectAltName=DNS:${HOST}" \
        -noenc -keyout "${SSLPFX}-key.pem" -out "${SSLPFX}-cert.pem" \
        2>> "${SSLLOG}" || die "Failed to create ${HOST} certificate"
        echo "Created ${HOST} certificate"
    fi
    SSLARG=("--certfile=\"${SSLPFX}-cert.pem\"" "--keyfile=\"${SSLPFX}-key.pem\"")
fi

(
    while [[ $(find "${RUNTIME_DIR}" -type f -name 'jpserver-*-open.html' 2>/dev/null | wc -l) = 0 ]]; do
        echo "Waiting for jpserver-*-open.html to appear"
        sleep 1
    done
    FILE="$(find "${RUNTIME_DIR}" -type f -name 'jpserver-*-open.html')"
    if [[ $(wc -l <<< ${FILE}) != 1 ]]; then
        CONCAT=$(xargs -r <<< ${FILE})
        echo "ERROR: got '${CONCAT}' for jpserver files"
        ls -R "${RUNTIME_DIR}"
        echo "ERROR: Multiple jpserver-*-open.html files"
        exit 1
    fi

    sleep 1 # reduce race with file being written
    REDIRECT="${BIND_DIR}/redirect.html"
    perl -ne 's,://\S+:8888/tree,://localhost:8888/tree,;print' < "${FILE}" >"${REDIRECT}"
    URL=$(perl -ne 'print $1 if m,url=(https?://localhost:8888/tree\S+)",;' <"${REDIRECT}")

    for i in {1..10}; do
        echo
        echo
        echo
        echo "Either:"
        echo "  a) Visit: ${URL}"
        echo "  b) open jupyter/bind_dir/redirect.html on your host machine"
        echo "  c) docker compose cp jupyter:${BIND_DIR}/redirect.html ."
        echo "      and open redirect.html in a browser"
        echo "  Note: the token is stable unless you delete docker_volume/jupyter_notebook_config.py"
        sleep 30
    done
) &

trap "kill $!" EXIT

cd "${WORK_DIR}"
poetry run jupyter notebook "${SSLARG[@]}" --no-browser --ip 0.0.0.0 "$@"
