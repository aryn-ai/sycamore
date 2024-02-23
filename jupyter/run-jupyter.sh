#!/bin/bash

die() {
    echo "ERROR:" "$@" >&2
    exit 1
}

mkdir -p $HOME/.jupyter
JUPYTER_CONFIG_DOCKER=/app/work/docker_volume/jupyter_notebook_config.py

if [[ ! -f "${JUPYTER_CONFIG_DOCKER}" ]]; then
    TOKEN=$(openssl rand -hex 24)
    cat >"${JUPYTER_CONFIG_DOCKER}" <<EOF
# Configuration file for notebook.

c = get_config()  #noqa

c.IdentityProvider.token = '${TOKEN}'
EOF
fi
ln -snf "${JUPYTER_CONFIG_DOCKER}" $HOME/.jupyter

rm /app/.local/share/jupyter/runtime/jpserver-*-open.html 2>/dev/null

if [[ ${SSL} == 0 ]]; then
    echo "Jupyter not serving over SSL."
    SSLARG=
else
    : ${HOST:=localhost}
    SSLPFX="/app/work/docker_volume/${HOST}"
    SSLLOG="/app/work/docker_volume/openssl.err"
    if [[ (! -f ${SSLPFX}-key.pem) || (! -f ${SSLPFX}-cert.pem) ]]; then
        openssl req -batch -x509 -newkey rsa:4096 -days 10000 \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=${HOST}" \
        -extensions v3_req -addext "subjectAltName=DNS:${HOST}" \
        -noenc -keyout "${SSLPFX}-key.pem" -out "${SSLPFX}-cert.pem" \
        2>> "${SSLLOG}" || die "Failed to create ${HOST} certificate"
        echo "Created ${HOST} certificate"
    fi
    SSLARG="--certfile=\"${SSLPFX}-cert.pem\" --keyfile=\"${SSLPFX}-key.pem\""
fi

(
    while [[ $(ls /app/.local/share/jupyter/runtime/jpserver-*-open.html 2>/dev/null | wc -w) = 0 ]]; do
        echo "Waiting for jpserver-*-open.html to appear"
        sleep 1
    done
    FILE="$(ls /app/.local/share/jupyter/runtime/jpserver-*-open.html)"
    if [[ $(echo "${FILE}" | wc -w) != 1 ]]; then
        echo "ERROR: got '${FILE}' for jpserver files"
        ls /app/.local/share/jupyter/runtime
        echo "ERROR: Multiple jpsterver-*-open.html files"
        exit 1
    fi

    sleep 1 # reduce race with file being written
    REDIRECT=/app/work/bind_dir/redirect.html
    perl -ne 's,://\S+:8888/tree,://localhost:8888/tree,;print' < "${FILE}" >"${REDIRECT}"
    URL=$(perl -ne 'print $1 if m,url=(https?://localhost:8888/tree\S+)",;' <"${REDIRECT}")

    for i in {1..10}; do
        echo
        echo
        echo
        echo "Either:"
        echo "  a) Visit: ${URL}"
        echo "  b) open jupyter/bind_dir/redirect.html on your host machine"
        echo "  c) docker compose cp jupyter:/app/work/bind_dir/redirect.html ."
        echo "      and open redirect.html in a broswer"
        echo "  Note: the token is stable unless you delete docker_volume/jupyter_notebook_config.py"
        sleep 30
    done
) &

cd /app/work
poetry run jupyter notebook ${SSLARG} --no-browser --ip 0.0.0.0 "$@"
