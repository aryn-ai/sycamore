#!/bin/bash
# TODO: https://github.com/aryn-ai/sycamore/issues/150 - detect low disk space and error out.
# on macos you fix it in docker desktop > settings > resources > scroll down > virtual disk limit
main() {
    local http_status_port=43477 # from rand
    local http_server_dir=/tmp/http_server
    local http_status_file="${http_server_dir}/statusz"
    [[ ! -f "${http_status_file}" ]] || rm -f "${http_status_file}"

    BASE_URL=http://localhost:9200
    ARYN_STATUSDIR=/usr/share/opensearch/data/aryn_status
    mkdir -p "${ARYN_STATUSDIR}"

    set -e
    # TODO: https://github.com/aryn-ai/sycamore/issues/151 - show aryn logs then opensearch

    LOG_BASE="${ARYN_STATUSDIR}/opensearch.log"
    if opensearch_up; then
        echo "OpenSearch appears to already be running, not starting it again"
    else
        echo "Should start opensearch"
        LOG_FILE="${LOG_BASE}.$(date +%Y-%m-%d--%H:%M:%S)"
        ln -snf "${LOG_FILE}" "${LOG_BASE}"
        ./opensearch-docker-entrypoint.sh >"${LOG_FILE}" 2>&1 &
        echo $! >/tmp/opensearch.pid
        trap "kill -TERM $(cat /tmp/opensearch.pid)" EXIT
        wait_or_die opensearch_up "opensearch to start"
    fi

    PERSISTENT_ENV="${ARYN_STATUSDIR}/persistent_env"
    [[ -f "${PERSISTENT_ENV}" ]] || setup_persistent
    source "${PERSISTENT_ENV}"

    setup_transient

    mkdir -p "${http_server_dir}"
    echo "OK" >"${http_status_file}"
    python3 -m http.server -d http_serve "${http_status_port}" -d "${http_server_dir}" \
            >/tmp/http_server.log 2>&1 &
    local pid=$!
    echo "${pid}" >/tmp/http_server.pid
    trap "kill -TERM $(cat /tmp/http_server.pid)" EXIT
    disown "${pid}" # don't wait for it to exit

    if [[ -z "${LOG_FILE}" ]]; then
        echo "Did not start opensearch, should exit shortly"
        echo "Opensearch log may be at ${LOG_BASE}"
    else
        echo "Waiting for opensearch to exit."
        echo "Log file: ${LOG_FILE}"
        echo "Also linked to: ${LOG_BASE}"
    fi
    wait
    exit 0
}

wait_or_die() {
    [[ ! -z "$1" ]] || die "Missing wait_for command"
    [[ ! -z "$2" ]] || die "Missing wait_for message"

    local max_reps=$3
    [[ -z "${max_reps}" ]] && max_reps=60

    echo -n "Waiting for $2..."
    local i
    for i in $(seq "${max_reps}"); do
        if "$1"; then
            echo " Success"
            return 0
        fi
        echo -n .
        sleep 1
    done
    echo " Failed!"
    die "$2 did not return true with $max_reps tries"
}

die() {
    echo "ERROR: " "$@" 1>&2
    exit 1
}

opensearch_up() {
    local file="${ARYN_STATUSDIR}/opensearch.status"
    rm "${file}" 2>/dev/null
    _curl "${BASE_URL}" -o "${file}" || return 1
    [[ -r "${file}" ]] || return 1
    local name="$(jq -r '.name' "${file}")"
    [[ -z "${name}" ]] && return 1
    return 0
}

_curl() {
    # Warning: some error output is suppressed by the -s
    /usr/bin/curl -s "$@"
}

setup_persistent() {
    sp_cluster_settings
    PERSISTENT_ENV_TMP="${PERSISTENT_ENV}.tmp"
    rm -f "${PERSISTENT_ENV}" "${PERSISTENT_ENV_TMP}" 2>/dev/null
    touch "${PERSISTENT_ENV_TMP}"
    sp_register_model_group
    # TODO: https://github.com/aryn-ai/sycamore/issues/152 - debug task id stability
    sp_setup_embedding_model
    sp_setup_openai_model

    sp_create_rag_pipeline
    sp_create_non_rag_pipeline

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
	"memory_feature_enabled": "true",
	"rag_pipeline_feature_enabled": "true",
	"only_run_on_ml_node": "false",
	"allow_registering_model_via_url": "true"
      }
    }
  }
}
END
    grep error "${file}" && die "Error setting cluster settings"
    echo "CLUSTER SETTINGS SET"
}

_curl_json() {
    _curl --header "Content-Type: application/json" --header "Accept:application/json" "$@"
}

sp_register_model_group() {
    local file="${ARYN_STATUSDIR}/curl.model_group"
    _curl_json -X POST "${BASE_URL}/_plugins/_ml/model_groups/_register" \
        -o "${file}" \
        --data @- <<END || die "Error registering model group"
{
  "name": "conversational_search_models",
  "description": "Public model group of the conversational search models we use"
}
END
    if false; then
        echo "---------------------------"
        cat "${file}"
        echo "---------------------------"
    fi
    # TODO: https://github.com/aryn-ai/sycamore/issues/152 - debug embedding task id stability
    local err='The name you provided is already being used by a model group with ID:'
    [[ "$(grep -c "${err}" <"${file}")" = 1 ]] && \
        die "model was already in use; something went wrong in setup"

    MODEL_GROUP_ID=$(jq -r '.model_group_id' "${file}")
    [[ -z "${MODEL_GROUP_ID}" || "${MODEL_GROUP_ID}" == "null" ]] &&  die "No model group ID"
    echo "MODEL_GROUP_ID='${MODEL_GROUP_ID}'" >>"${PERSISTENT_ENV_TMP}"
}

sp_setup_embedding_model() {
    local all_config=$(jq '@json' <<END
{
  "_name_or_path": "nreimers/MiniLM-L6-H384-uncased",
  "architectures": [
    "BertModel"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 384,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "layer_norm_eps": 1E-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.8.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
END
)

    local file="${ARYN_STATUSDIR}/curl.embedding_model"
    _curl_json -X POST "${BASE_URL}/_plugins/_ml/models/_register" \
          -o "${file}" \
          --data @- <<END || die "Error registering embedding model"
{
  "name": "all-MiniLM-L6-v2",
  "version": "1.0.0",
  "description": "embedding model",
  "model_format": "TORCH_SCRIPT",
  "model_group_id": "${MODEL_GROUP_ID}",
  "model_content_hash_value": "c15f0d2e62d872be5b5bc6c84d2e0f4921541e29fefbef51d59cc10a8ae30e0f",
  "model_config": {
    "model_type": "bert",
    "embedding_dimension": 384,
    "framework_type": "sentence_transformers",
    "all_config": ${all_config}
  },
  "url": "https://artifacts.opensearch.org/models/ml-models/huggingface/sentence-transformers/all-MiniLM-L6-v2/1.0.1/torch_script/sentence-transformers_all-MiniLM-L6-v2-1.0.1-torch_script.zip"
}
END

    local id=$(jq -r '.task_id' "${file}")
    [[ -z "${id}" || "${id}" == "null" ]] && die "No embedding task ID"
    EMBEDDING_TASK_ID="${id}"
    echo "EMBEDDING_TASK_ID='${EMBEDDING_TASK_ID}'" >>"${PERSISTENT_ENV_TMP}"

    deploy_model "" "${EMBEDDING_TASK_ID}" "embedding"
    EMBEDDING_MODEL_ID="${MODEL_ID}"
    echo "EMBEDDING_MODEL_ID=${EMBEDDING_MODEL_ID}" >>"${PERSISTENT_ENV_TMP}"
}

sp_setup_openai_model() {
    [[ -z "${OPENAI_API_KEY}" ]] && die "No OPENAI_API_KEY set"
    wait_or_die sp_create_openai_connector 'creation of OpenAI connector' 60
    sp_register_openai_model
    deploy_model "" "${OPENAI_TASK_ID}" "OpenAI"
    OPENAI_MODEL_ID="${MODEL_ID}"
    echo "OPENAI_MODEL_ID=${OPENAI_MODEL_ID}" >>"${PERSISTENT_ENV_TMP}"
}

sp_create_openai_connector() {
    local file="${ARYN_STATUSDIR}/curl.openai_connector"

    local RB0='{
"model":"${parameters.model}",
"messages":${parameters.messages},
"temperature":${parameters.temperature}
}'
    local RB1="${RB0//\"/\\\"}"
    local REQBODY="${RB1//$'\n'/ }"

    _curl_json -X POST "${BASE_URL}/_plugins/_ml/connectors/_create" \
          -o "${file}" \
          --data @- <<END
{
  "name": "OpenAI Chat Connector",
  "description": "The connector to public OpenAI model service for GPT 3.5",
  "version": 2,
  "protocol": "http",
  "parameters": {
    "endpoint": "api.openai.com",
    "model": "gpt-3.5-turbo",
    "temperature": 0
  },
  "credential": {
    "openAI_key": "${OPENAI_API_KEY}"
  },
  "actions": [
    {
	"action_type": "predict",
	"method": "POST",
	"url": "https://\${parameters.endpoint}/v1/chat/completions",
	"headers": {
	  "Authorization": "Bearer \${credential.openAI_key}"
	},
	"request_body": "${REQBODY}"
    }
  ]
}
END

    local id="$(jq -r '.connector_id' "${file}")"
    [[ -z "${id}" || "${id}" == 'null' ]] && return 1
    OPENAI_CONN_ID="${id}"
    echo "OPENAI_CONN_ID=${OPENAI_CONN_ID}" >>"${PERSISTENT_ENV_TMP}"
    return 0
}

sp_register_openai_model() {
    local file="${ARYN_STATUSDIR}/curl.openai_model"
    _curl_json -X POST "${BASE_URL}/_plugins/_ml/models/_register" \
               -o "${file}" \
               --data @- <<END || die "Error registering GPT model"
{
  "name": "openAI-gpt-3.5-turbo",
  "function_name": "remote",
  "description": "gpt model",
  "connector_id": "${OPENAI_CONN_ID}",
  "model_group_id": "${MODEL_GROUP_ID}"
}
END

    local id="$(jq -r '.task_id' "${file}")"
    [[ -z "${id}" ]] && die "No OpenAI task ID from registering model"

    OPENAI_TASK_ID="${id}"
    echo "OPENAI_TASK_ID=${OPENAI_TASK_ID}" >>"${PERSISTENT_ENV_TMP}"
}

deploy_model() {
    local model_id="$1" # if empty will be derived from task
    local task_id="$2"
    local name="$3"
    [[ -z "${task_id}" ]] && die "ERROR missing task_id"
    [[ -z "${name}" ]] && die "ERROR missing name"

    if [[ -z "${model_id}" ]]; then
        wait_task "${task_id}" "$name model to become ready"
        model_id="${GET_TASK_MODEL_ID}"
        echo "Fetched $name model id $model_id for task_id $task_id"
    else
        echo "Using existing model id ${model_id}"
    fi
    MODEL_ID="${model_id}"
    DEPLOY_MODEL_LOG_FILE="${ARYN_STATUSDIR}/curl.deploy_model_task.${name}"
    wait_or_die deploy_model_try "deploy of model ${MODEL_ID}, task ${task_id}, name ${name}" 60
    wait_task "${DEPLOY_MODEL_DEP_TASK_ID}" "$name model to deploy"
    # TODO: https://github.com/aryn-ai/sycamore/issues/153 - debug task waiting
}

wait_task() {
    local task_id="$1"
    local msg="$2"

    [[ -z "${task_id}" ]] && die "missing task_id"
    [[ -z "${msg}" ]] && die "missing msg"

    GET_TASK_ID="${task_id}"
    wait_or_die get_task "${msg}" 60
}

get_task() {
    local file="${ARYN_STATUSDIR}/curl.get_task"
    rm -f "${file}" 2>/dev/null
    _curl "${BASE_URL}/_plugins/_ml/tasks/${GET_TASK_ID}" -o "${file}"

    [[ -r "${file}" ]] || return 1
    local model="$(jq -r '.model_id' "${file}")"
    [[ -z "${model}" || "${model}" = 'null' ]] && return 1

    GET_TASK_MODEL_ID="${model}"
    return 0
}

deploy_model_try() {
    _curl -X POST "${BASE_URL}/_plugins/_ml/models/${MODEL_ID}/_deploy" \
          -o "${DEPLOY_MODEL_LOG_FILE}"
    local dep_task_id="$(jq -r '.task_id' "${DEPLOY_MODEL_LOG_FILE}")"
    [[ -z "${dep_task_id}" || "${dep_task_id}" == null ]] \
        && return 1
    DEPLOY_MODEL_DEP_TASK_ID="${dep_task_id}"
}

sp_create_rag_pipeline() {
    _curl_json -X PUT "${BASE_URL}/_search/pipeline/hybrid_rag_pipeline" \
          -o "${ARYN_STATUSDIR}/curl.create_rag_pipeline" \
          --data @- <<END || die "Error registering hybrid RAG pipeline"
{
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.111, 0.889]
          }
        }
      }
    }
  ],
  "response_processors": [
    {
      "retrieval_augmented_generation": {
        "tag": "openai_pipeline",
        "description": "Pipeline Using OpenAI Connector",
        "model_id": "${OPENAI_MODEL_ID}",
        "context_field_list": ["text_representation"]
      }
    }
  ]
}
END
}

sp_create_non_rag_pipeline() {
    _curl_json -X PUT "${BASE_URL}/_search/pipeline/hybrid_pipeline" \
               -o "${ARYN_STATUSDIR}/curl.create_non_rag_pipeline" \
               --data @- <<END || die "Error registering non-RAG pipeline"
{
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.111, 0.889]
          }
        }
      }
    }
  ]
}
END
}

setup_transient() {
    deploy_model "${EMBEDDING_MODEL_ID}" "${EMBEDDING_TASK_ID}" "embedding"
    deploy_model "${OPENAI_MODEL_ID}" "${OPENAI_TASK_ID}" "OpenAI"
}

main
