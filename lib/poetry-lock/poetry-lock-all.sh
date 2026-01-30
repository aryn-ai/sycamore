#!/bin/bash

root="$(pwd)"
fail() {
    echo FAILED FAILED FAILED FAILED
    echo "$@"
    echo FAILED FAILED FAILED FAILED
    exit 1
}
[[ -f "${root}/lib/sycamore/pyproject.toml" ]] || fail "run in root git directory"

INSTALLOUT=/tmp/poetry-install.$$.out
DOWNGRADED=/tmp/downgraded.$$.out
function cleanup {
    rm -f "${INSTALLOUT}" "${DOWNGRADED}"
}
trap cleanup EXIT

echo "Setting in-project virtualenvs locally"
poetry config virtualenvs.in-project true --local
poetry install --no-root
[[ -d .venv ]] || fail "Did not get .venv directory after install"

tomls=$(find . ! -path "**/site-packages/**/*" -name pyproject.toml)
for i in ${tomls}; do
    dest="$(dirname $i)"
    [[ "$dest" == "." ]] && continue
    ln -snf "$(pwd)"/.venv "$dest"/.venv
    [[ -L "$dest"/.venv ]] || fail "$dest/.venv isn't symlink"
done

for i in ${tomls}; do
    (
        echo "--------------------- processing in $i"
        cd $(dirname "$i")
        poetry lock || fail "broke on regular case 'poetry lock' $i"
        poetry install --no-root 2>&1 | tee "${INSTALLOUT}" || fail "broke on 'poetry install' for $i"
	grep 'Downgrading ' "${INSTALLOUT}" | awk '{print$3,$4}' | tr -d '(' > "${DOWNGRADED}"
	echo "* Downgraded from ${i}:"
        sort "${DOWNGRADED}"
        # [[ $(wc -l < "${DOWNGRADED}") -eq 0 ]] || fail "broke on downgrading $i it seems these packages are incompatible"
    ) || fail "broke on regular case $i"
done
echo "SUCCESS"
