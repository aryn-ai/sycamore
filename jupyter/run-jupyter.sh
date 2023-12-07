#!/bin/bash
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
    perl -ne 's,http://\S+:8888/tree,http://localhost:8888/tree,;print' < "${FILE}" >"${REDIRECT}"
    URL=$(perl -ne 'print $1 if m,url=(http://localhost:8888/tree\S+)",;' <"${REDIRECT}")

    for i in $(seq 10); do
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
poetry run jupyter notebook --no-browser --ip 0.0.0.0 "$@"
