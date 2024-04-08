#!/bin/bash

main() {
  echo "Get the newest git commits" >&2
  checkout_main_if_new
  local should_run=$?
  if [[ $should_run ]]; then
    echo "Changes detected. Running Tests" >&2
    poetry install
    build_containers
    poetry run pytest apps/integration
    handle_outputs
  else
    echo "No changes detected. Skipping integration tests" >&2
  fi
}

checkout_main_if_new() {
  old_sha="$(git rev-parse HEAD)"
  git fetch origin main >&2
  new_sha="$(git rev-parse FETCH_HEAD)"
  if [[ "${old_sha}" != "${new_sha}" ]]; then
    git pull origin main >&2
    return 0
  else
    return 1
  fi
}

build_containers() {
  echo "Yep, definitely building containers. That's what this function does" >&2
}

handle_outputs() {
  echo "Yep, definitely handling test outputs. That's what this function does" >&2
}


main
