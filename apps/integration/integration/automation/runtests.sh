#!/bin/bash

main() {
  if [[ ! -d ".git" ]]; then
    echo "Error: please run this script from sycamore root!" >&2
    exit 1
  fi
  echo "Get the newest git commits" >&2
  checkout_main_if_new
  local should_run=$?
  [[ "$1" == "--force" ]] && should_run=0
  if [[ $should_run == 0 ]]; then
    echo "Changes detected. Running Tests" >&2
    poetry install
    build_containers
    runtests
    handle_outputs
  else
    echo "No changes detected. Skipping integration tests. Use $0 --force to force" >&2
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

runtests() {
  docker system prune -f --volumes
  docker compose up reset
  poetry run pytest apps/integration/ -p integration.conftest --noconftest --docker-tag latest_rc
  # this is a complicated command, so: ^                        ^            ^ test against containers tagged latest_rc
  #                                    |                     don't load conftest at pytest runtime; it's already loaded
  #                                     load conftest with plugins, to capture the custom command line arg --docker-tag
}


main "$@"
