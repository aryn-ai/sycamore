#!/bin/bash

TAG="$1"
[[ -z "${TAG}" ]] && TAG="latest_rc"

main() {
  if [[ ! -d ".git" ]]; then
    echo "Error: please run this script from sycamore root!" >&2
    exit 1
  fi
  echo "Building/testing tag ${TAG}" >&2
  echo "Get the newest git commits" >&2
  checkout_main_if_new
  local should_run=$?
  if [[ $should_run ]]; then
    echo "Changes detected. Running Tests" >&2
    poetry install
    build_containers
    runtests
    handle_outputs
  else
    echo "No changes detected. Skipping integration tests" >&2
  fi
}

error() {
  echo "ERROR: $@"
  exit 1
}

checkout_main_if_new() {
  old_sha="$(git rev-parse HEAD)"
  git fetch origin main >&2
  new_sha="$(git rev-parse FETCH_HEAD)"
  if [[ "${old_sha}" != "${new_sha}" ]]; then
    git pull --rebase origin main >&2
    return 0
  else
    return 1
  fi
}

build_containers() {
  echo "Yep, definitely building containers. That's what this function does" >&2
  docker-build-hub apps/crawler/crawler/http/Dockerfile
  docker-build-hub apps/crawler/crawler/s3/Dockerfile
  docker-build-hub apps/importer/Dockerfile.buildx
  docker-build-hub apps/opensearch/Dockerfile
  docker-build-hub apps/jupyter/Dockerfile.buildx --build-arg=TAG="${TAG}"
  docker-build-hub apps/demo-ui/Dockerfile.buildx
  docker-build-hub apps/remote-processor-service/Dockerfile.buildx
}

handle_outputs() {
  echo "Yep, definitely handling test outputs. That's what this function does" >&2
}

runtests() {
  docker system prune -f --volumes
  docker compose up reset
  poetry run pytest apps/integration/ -p integration.conftest --noconftest --docker-tag "${TAG}"
  # this is a complicated command, so: ^                        ^            ^ test against containers tagged latest_rc
  #                                    |                     don't load conftest at pytest runtime; it's already loaded
  #                                     load conftest with plugins, to capture the custom command line arg --docker-tag
}

docker-build-hub() {
  local docker_file="$1"
  [[ -n "${docker_file}" ]] || error "missing ${docker_file}"
  local repo_name="$(_docker-repo-name "${docker_file}")"
  [[ -n "${repo_name}" ]] || error "empty repo name"
  shift

  local platform=linux/amd64,linux/arm64
  echo
  echo "Building in sycamore and pushing to docker hub with repo name '${repo_name}'"
  docker buildx build "$(_docker-build-args)" -t "${repo_name}:${TAG}" -f "${docker_file}" --platform ${platform} \
     --cache-to type=registry,ref="${repo_name}:build-cache",mode=max \
     --cache-from type=registry,ref="${repo_name}:build-cache" \
     "$@" --push . || error "buildx failed"
  echo "Successfully built in $dir using docker file $docker_file"
}

_docker-repo-name() {
  local docker_file="$1"
  echo "Finding repo name in: ${docker_file}" >&2
  local repo_name="$(grep '^# Repo name: ' "${docker_file}" | awk '{print $4}')"
  if (( $(wc -w <<< ${repo_name}) != 1 )); then
    echo "Unable to find repo name in ${docker_file}" 1>&2
    exit 1
  fi
  echo "${repo_name}"
}

_docker-build-args() {
  local branch="$(git status | head -1 | grep 'On branch ' | awk '{print $3}')"
  local rev="$(git rev-parse --short HEAD)"
  local date="$(git show -s --format=%ci HEAD | sed 's/ /_/g')"
  local diff=unknown
  if [[ $(git status | grep -c 'nothing to commit, working tree clean') = 1 ]]; then
    diff=clean
  else
    diff="pending_changes_$(git diff HEAD | shasum | awk '{print $1}')"
  fi
  echo "--build-arg=GIT_BRANCH=${branch} --build-arg=GIT_COMMIT=${rev}--${date} --build-arg=GIT_DIFF=${diff}"
}

main
