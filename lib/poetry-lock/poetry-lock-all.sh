#!/bin/bash

root="$(pwd)"
fail() {
    echo "$@"
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
    [[ $i = *openai-proxy* ]] && continue
    (
        echo "--------------------- processing in $i"
        cd $(dirname "$i")
        poetry lock --no-update || exit 1
        poetry install 2>&1 | tee /tmp/poetry-install.out || exit 1
        perl -ne 'print qq{$1 = "$2"\n} if /Downgrading (\S+) \((\S+) ->/o;' </tmp/poetry-install.out
    )
done
