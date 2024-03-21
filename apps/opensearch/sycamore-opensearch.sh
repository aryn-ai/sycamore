#!/bin/bash
echo "Version-Info, Aryn Opensearch Branch: ${GIT_BRANCH}"
echo "Version-Info, Aryn Opensearch Commit: ${GIT_COMMIT}"
echo "Version-Info, Aryn Opensearch Diff: ${GIT_DIFF}"
echo "Version-Info, Aryn Opensearch Architecture: $(uname -m)"

# TODO: https://github.com/aryn-ai/sycamore/issues/150 - detect low disk space and error out.
# on macos you fix it in docker desktop > settings > resources > scroll down > virtual disk limit
main() {
    BASE_URL=https://localhost:9200
    ARYN_STATUSDIR=/usr/share/opensearch/data/aryn_status
    mkdir -p "${ARYN_STATUSDIR}"

    set -e
    # TODO: https://github.com/aryn-ai/sycamore/issues/151 - show aryn logs then opensearch

    LOG_BASE="${ARYN_STATUSDIR}/opensearch.log"
    if opensearch_up_ssl; then
        echo "OpenSearch appears to already be running, not starting it again"
    elif opensearch_up_insecure; then
        die "OpenSearch appears insecure"
    else
        echo "Should start opensearch"
        LOG_FILE="${LOG_BASE}.$(date +%Y-%m-%d--%H:%M:%S)"
        ln -snf "${LOG_FILE}" "${LOG_BASE}"
        create_certificates
        ./opensearch-docker-entrypoint.sh >"${LOG_FILE}" 2>&1 &
        echo $! >/tmp/opensearch.pid
        trap "kill -TERM $(cat /tmp/opensearch.pid)" EXIT
        wait_or_die opensearch_up_net "opensearch to start" 300
        setup_security
    fi

    flick_rag_feature

    PERSISTENT_ENV="${ARYN_STATUSDIR}/persistent_env"
    [[ -f "${PERSISTENT_ENV}" ]] || setup_persistent
    source "${PERSISTENT_ENV}"

    setup_transient

    if [[ -z "${LOG_FILE}" ]]; then
        echo "Did not start opensearch, should exit shortly"
        echo "Opensearch log may be at ${LOG_BASE}"
    else
        echo "Waiting for opensearch to exit."
        echo "Log file: ${LOG_FILE}"
        echo "Also linked to: ${LOG_BASE}"
    fi
    wait
    if [[ ${NOEXIT} -gt 0 ]]; then
        echo "Not exiting due to NOEXIT.  Feel free to poke around container."
        sleep inf
    fi
    exit 0
}

wait_or_die() {
    [[ ! -z "$1" ]] || die "Missing wait_for command"
    [[ ! -z "$2" ]] || die "Missing wait_for message"

    local max_reps=$3
    [[ -z "${max_reps}" ]] && max_reps=60

    local i
    for i in $(seq "${max_reps}"); do
        if "$1"; then
            echo "Waiting for $2... Success"
            return 0
        fi
        echo "Waiting for $2... Sleeping. Try $i/${max_reps}"
        sleep 1
    done
    die "$2 did not happen within $max_reps seconds"
}

die() {
    echo "ERROR:" "$@" 1>&2
    if [[ ${NOEXIT} -gt 0 ]]; then
        echo "Not dying due to NOEXIT.  Feel free to poke around container."
        sleep inf
    fi
    exit 1
}

debug() { # <msg> <file>
    if [[ ${DEBUG} -gt 0 ]]; then
        local i=0 j n stack
        (( n = ${#FUNCNAME[@]} - 2 ))
        while (( i < n )); do
            (( j = i + 1 ))
            stack="${FUNCNAME[$j]}:${BASH_LINENO[$i]} ${stack}"
            (( i = j ))
        done
        echo "DEBUG ${stack}$1" >&2
        if [[ -s $2 ]]; then
            echo 'vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv' >&2
            awk 1 $2 >&2 # 'awk 1' ensures a newline at the end
            echo '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^' >&2
        fi
    fi
}

info() {
    echo "INFO $@" >&2
}

opensearch_up_insecure() {
    local out=$(_curl http://localhost:9200/)
    [[ -z ${out} ]] && return 1
    debug "${out}"
    local name=$(jq -r '.name' <<< ${out})
    [[ -z ${name} ]] && return 1
    return 0
}

opensearch_up_ssl() {
    local out=$(_curl https://localhost:9200/)
    [[ -z ${out} ]] && return 1
    debug "${out}"
    [[ ${out} =~ "not initialized" ]] && return 1
    local name=$(jq -r '.name' <<< ${out})
    [[ -z ${name} ]] && return 1
    return 0
}

opensearch_up_net() {
    # This checks for any activity at the host:port.  No error (zero return)
    # is obviously good, but 52, meaning empty response, is also indicative.
    _curl http://localhost:9200/ -o /dev/null
    [[ ($? != 0) && ($? != 52) ]] && return 1
    return 0
}

_curl() {
    # Warning: some error output is suppressed by --silent
    # We use --insecure due to self-signed certificates
    /usr/bin/curl --insecure --silent "$@"
}

setup_persistent() {
    sp_cluster_settings
    PERSISTENT_ENV_TMP="${PERSISTENT_ENV}.tmp"
    rm -f "${PERSISTENT_ENV}" "${PERSISTENT_ENV_TMP}" 2>/dev/null
    touch "${PERSISTENT_ENV_TMP}"

    initialize_message_index

    python3 setup_models.py || die "Failed to setup models"

    mv "${PERSISTENT_ENV_TMP}" "${PERSISTENT_ENV}"
    echo "Setup Persistent env:"
    cat "${PERSISTENT_ENV}"
}

sp_cluster_settings() {
    local file="${ARYN_STATUSDIR}/curl.cluster_settings"
    _curl_json -X PUT "${BASE_URL}/_cluster/settings" \
         -o "${file}" --data @- <<END || die "Error in cluster settings"
{
  "persistent": {
    "plugins": {
      "ml_commons": {
	"only_run_on_ml_node": "false",
	"allow_registering_model_via_url": "true",
	"native_memory_threshold": 100
      }
    }
  }
}
END

    debug "" "${file}"
    grep error "${file}" && die "Error setting cluster settings"
    echo "CLUSTER SETTINGS SET"
}

_curl_json() {
    _curl --header "Content-Type: application/json" --header "Accept:application/json" "$@"
}


flick_rag_feature() {
    # This exists because the rag_pipeline_feature_enabled setting
    # only recognizes when it changes. Since it starts in the 'on'
    # position now, we have to turn it off and then on again
    _curl_json -X PUT "${BASE_URL}/_cluster/settings" \
          -o "${ARYN_STATUSDIR}/curl.disable_rag" \
          --data @- <<END || die "Error in cluster settings"
{
  "persistent": {
    "plugins.ml_commons.rag_pipeline_feature_enabled": "false"
  }
}
END
    debug "" "${ARYN_STATUSDIR}/curl.disable_rag"
    _curl_json -X PUT "${BASE_URL}/_cluster/settings" \
          -o "${ARYN_STATUSDIR}/curl.reenable_rag" \
          --data @- <<END || die "Error in cluster settings"
{
  "persistent": {
    "plugins.ml_commons.rag_pipeline_feature_enabled": "true"
  }
}
END
    debug "" "${ARYN_STATUSDIR}/curl.reenable_rag"
}

initialize_message_index() {
    # create a dummy conversation, dummy interaction and then
    # delete them so that the demo ui doesn't throw a fit trying
    # to load interactions from an index that doesn't exist
    _curl_json -XPOST "${BASE_URL}/_plugins/_ml/memory" \
          -o "${ARYN_STATUSDIR}/curl.create_memory" \
          --data @- <<END || die "Error creating dummy conversation"
{
  "name": "ignoreme"
}
END
    debug "" "${ARYN_STATUSDIR}/curl.create_memory"

    local convo_id=$(jq -r ".memory_id" "${ARYN_STATUSDIR}/curl.create_memory")
    _curl_json -XPOST "${BASE_URL}/_plugins/_ml/memory/${convo_id}/messages" \
          -o "${ARYN_STATUSDIR}/curl.create_message" \
          --data @- <<END || die "Error creating dummy message"
{
  "input": "dummy",
  "prompt_template": "dummy",
  "response": "dummy",
  "origin": "dummy"
}
END
    debug "" "${ARYN_STATUSDIR}/curl.create_message"

    _curl_json -XDELETE "${BASE_URL}/_plugins/_ml/memory/${convo_id}" \
          -o "${ARYN_STATUSDIR}/curl.delete_memory" || die "Error deleting dummy memory"

    debug "" "${ARYN_STATUSDIR}/curl.delete_memory"
}

setup_transient() {
    # Make sure OpenSearch isn't doing something wacky...
    _curl "${BASE_URL}/_cluster/settings" \
    | grep -Fq aryn_deploy_complete && die "aryn_deploy_complete already set"

    python3 setup_models.py || die "Failed to setup models"

    _curl -X PUT "${BASE_URL}/_cluster/settings" -o /dev/null --json \
    '{"transient":{"cluster":{"metadata":{"aryn_deploy_complete":1}}}}'
}

with_env_update() {
    local env_updating_function="$1"
    read_persistent_env
    "$env_updating_function"
    write_persistent_env
}

read_persistent_env() {
    PERSISTENT_ENV_TMP="${PERSISTENT_ENV}.tmp"
    rm -f "${PERSISTENT_ENV_TMP}" 2>/dev/null
    cp "${PERSISTENT_ENV}" "${PERSISTENT_ENV_TMP}"
}

write_persistent_env() {
    mv "${PERSISTENT_ENV_TMP}" "${PERSISTENT_ENV}"
}

create_certificates() {
    local HOST="${SSL_HOSTNAME:-localhost}"
    local DAYS=10000
    local LOG="${ARYN_STATUSDIR}/openssl.err"
    truncate -s 0 "${LOG}"

    # 1. Make fake certificate authority (CA) certificate.  OpenSearch
    # requires a root certificate to be specified.
    if [[ (! -f data/cakey.pem) || (! -f data/cacert.pem) ]]; then
        openssl req -batch -x509 -newkey rsa:4096 -days "${DAYS}" \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=Fake CA" \
        -extensions v3_ca -noenc -keyout data/cakey.pem -out data/cacert.pem \
        2>> "${LOG}" || die "Failed to create CA certificate"
        echo "Created CA certificate"
    fi

    # 2a. Create certificate signing request (CSR) for the node certificate.
    if [[ (! -f data/node-key.pem) || (! -f data/node-cert.pem) ]]; then
        openssl req -batch -newkey rsa:4096 \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=${HOST}" \
        -extensions v3_req -addext "basicConstraints=critical,CA:FALSE" \
        -addext "subjectAltName=DNS:${HOST}" \
        -noenc -keyout data/node-key.pem -out data/node-req.pem \
        2>> "${LOG}" || die "Failed to create node CSR"

        # 2b. Use the fake CA to sign the node CSR, yielding a certificate.
        openssl x509 -req -CA data/cacert.pem -CAkey data/cakey.pem \
        -copy_extensions copy -days "${DAYS}" \
        -in data/node-req.pem -out data/node-cert.pem \
        2>> "${LOG}" || die "Failed to create node certificate"
        echo "Created node certificate"
    fi

    # 3a. Create certificate signing request (CSR) for the admin certificate.
    if [[ (! -f data/admin-key.pem) || (! -f data/admin-cert.pem) ]]; then
        openssl req -batch -newkey rsa:4096 \
        -subj "/C=US/ST=California/O=Aryn.ai/CN=Admin" \
        -extensions v3_req -addext "basicConstraints=critical,CA:FALSE" \
        -noenc -keyout data/admin-key.pem -out data/admin-req.pem \
        2>> "${LOG}" || die "Failed to create admin CSR"

        # 3b. Use the fake CA to sign the admin CSR, yielding a certificate.
        openssl x509 -req -CA data/cacert.pem -CAkey data/cakey.pem \
        -copy_extensions copy -days "${DAYS}" \
        -in data/admin-req.pem -out data/admin-cert.pem \
        2>> "${LOG}" || die "Failed to create admin certificate"
        echo "Created admin certificate"
    fi

    rm -f data/node-req.pem data/admin-req.pem
    for X in cakey.pem cacert.pem node-key.pem node-cert.pem admin-key.pem admin-cert.pem; do
        chmod 600 "data/${X}"
        ln -sfn "../data/${X}" "config/${X}"
    done
    debug "" "${LOG}"
}

setup_security() {
    # Set up security plugin configuration as described here:
    # https://opensearch.org/docs/latest/security/configuration/security-admin/
    plugins/opensearch-security/tools/securityadmin.sh \
    -cd config/opensearch-security -icl -nhnv -cacert config/cacert.pem \
    -cert config/admin-cert.pem -key config/admin-key.pem

    # Wait for eventual consistency of changes to security index
    wait_or_die opensearch_up_ssl "opensearch on ssl" 30

    # Semaphore for SSL setup.  Useful for debugging and other scripts.
    _curl -X PUT "${BASE_URL}/_cluster/settings" -o /dev/null --json \
    '{"persistent":{"cluster":{"metadata":{"aryn_ssl_setup_complete":1}}}}'
}

main
