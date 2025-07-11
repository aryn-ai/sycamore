#!/bin/bash

root="$(pwd)"
fail() {
    echo FAILED FAILED FAILED FAILED
    echo "$@"
    echo FAILED FAILED FAILED FAILED
    exit 1
}
[[ -f "${root}/lib/sycamore/pyproject.toml" ]] || fail "run in root git directory"
echo "Setting in-project virtualenvs locally"
poetry config virtualenvs.in-project true --local
poetry install
[[ -d .venv ]] || fail "Did not get .venv directory after install"

tomls=$(find . ! -path "**/site-packages/**/*" -name pyproject.toml)
for i in ${tomls}; do
    dest="$(dirname $i)"
    [[ "$dest" == "." ]] && continue
    ln -snf "$(pwd)"/.venv "$dest"/.venv
    [[ -L "$dest"/.venv ]] || fail "$dest/.venv isn't symlink"
done

for i in ${tomls}; do
    # Demo UI needs 0.28, rest of sycamore needs 1.x
    if [[ $i = *openai-proxy* ]]; then
        (
            echo "--------------------- special casing in $i"
            cd $(dirname "$i")
            poetry lock || fail "broke on special case 'poetry lock' for $i"
            # Do not apply the consistency logic, it can't do anything useful.
        ) || fail "broke on special case for $i"
        continue
    fi
    (
        echo "--------------------- processing in $i"
        cd $(dirname "$i")
        poetry lock || fail "broke on regular case 'poetry lock' $i"
        poetry install 2>&1 | tee /tmp/poetry-install.out || fail "broke on 'poetry install' for $i"
        perl -ne 'print qq{$1 = "$2"\n} if /Downgrading (\S+) \((\S+) ->/o;' </tmp/poetry-install.out >/tmp/downgraded
        cat /tmp/downgraded
        [[ $(wc -l </tmp/downgraded) -eq 0 ]] || fail "broke on downgrading $i it seems these packages are incompatible"
    ) || fail "broke on regular case $i"
done
echo "SUCCESS"
