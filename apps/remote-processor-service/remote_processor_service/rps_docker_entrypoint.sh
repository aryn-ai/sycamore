#!/bin/bash

main() {
    create_certificates
    poetry run server config/pipelines.yml --keyfile config/rps-key.pem --certfile config/rps-cert.pem
}

die() {
    echo "ERROR:" "$@" 1>&2
    if [[ ${NOEXIT} -gt 0 ]]; then
        echo "Not dying due to NOEXIT.  Feel free to poke around container."
        sleep inf
    fi
    exit 1
}

create_certificates() {
    local HOST="${SSL_HOSTNAME:-localhost}"
    local DAYS=10000
    local LOG="config/openssl.err"
    truncate -s 0 "${LOG}"

    # Create RPS certificate.
    if [[ (! -f config/rps-key.pem) || (! -f config/rps-cert.pem) ]]; then
        openssl req -batch -x509 -newkey rsa:4096 -days "${DAYS}" \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=${HOST}" \
        -extensions v3_req -addext "subjectAltName=DNS:${HOST}" \
        -noenc -keyout "config/rps-key.pem" -out "config/rps-cert.pem" 2> /dev/null
        echo "Created RPS certificate"
    fi

    for X in rps-key.pem rps-cert.pem; do
        chmod 600 "config/${X}"
    done
}

main
