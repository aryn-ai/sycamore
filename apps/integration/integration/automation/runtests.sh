#!/bin/bash

TAG="$1"
[[ -z "${TAG}" ]] && TAG="integration_tests"

NOW="$(date +"%Y-%m-%d_%H_%M_%S")"
ARCH="amd64"
[[ "$(uname -m)" = "arm64" ]] && ARCH="arm64"

RUNDIR="apps/integration/runs/${NOW}"
GIT_LOGFILE="${RUNDIR}/git.log"
DOCKER_LOGFILE="${RUNDIR}/docker.log"
POETRY_LOGFILE="${RUNDIR}/poetry.log"
PYTEST_LOGFILE="${RUNDIR}/pytest.log"
QUERY_LOGFILE="${RUNDIR}/test_queries.log"

main() {
  [[ ! -d ".git" ]] && die "Please run this script from sycamore root!"
  mkdir -p "${RUNDIR}"
  echo "Building/testing tag ${TAG}" >&2
  echo "Get the newest git commits" >&2
  if [[ $(checkout_main_if_new) ]]; then
    echo "Changes detected. Running Tests" >&2
    poetry install > "${POETRY_LOGFILE}" 2>&1 \
    && build_images > "${DOCKER_LOGFILE}" 2>&1 \
    && runtests > "${PYTEST_LOGFILE}" 2>&1
    local passed_tests=$?
    push_images >> "${DOCKER_LOGFILE}" 2>&1
    handle_outputs $passed_tests
  else
    echo "No changes detected. Skipping integration tests" >&2
  fi
}

error() {
  echo "ERROR: $@" >&2
}

checkout_main_if_new() {
  old_sha="$(git rev-parse HEAD)"
  git fetch origin main > "${GIT_LOGFILE}" 2>&1
  new_sha="$(git rev-parse FETCH_HEAD)"
  if [[ "${old_sha}" != "${new_sha}" ]]; then
    [[ $(git status | grep -c 'nothing to commit, working tree clean') = 1 ]] \
      || { echo "Working tree not clean" > "${GIT_LOGFILE}" && return 1; }
    git pull --rebase origin main >> "${GIT_LOGFILE}" 2>&1
    return 0
  else
    return 1
  fi
}

build_images() {
  echo "Building all images" >&2
  docker-build-hub apps/crawler/crawler/http/Dockerfile \
  && docker-build-hub apps/crawler/crawler/s3/Dockerfile \
  && docker-build-hub apps/importer/Dockerfile.buildx \
  && docker-build-hub apps/opensearch/Dockerfile \
  && docker-build-hub apps/jupyter/Dockerfile.buildx --build-arg=TAG="${TAG}" \
  && docker-build-hub apps/demo-ui/Dockerfile.buildx \
  && docker-build-hub apps/remote-processor-service/Dockerfile.buildx \
  && return 0
  return 1
}

handle_outputs() {
  echo "Handling test outputs" >&2
  local passed_tests="$1"
  mv test-output.log "${QUERY_LOGFILE}"
  [[ ${passed_tests} = 0 ]] && touch "${RUNDIR}/passed"
  [[ ${passed_tests} != 0 ]] && touch "${RUNDIR}/failed"
  aws s3 cp -r "${RUNDIR}" "s3://sycamore-ci/${ARCH}"
}

push_images() {
  echo "Pushing tested images to dockerhub" >&2
  docker-push-hub apps/crawler/crawler/http/Dockerfile \
  && docker-push-hub apps/crawler/crawler/s3/Dockerfile \
  && docker-push-hub apps/importer/Dockerfile.buildx \
  && docker-push-hub apps/opensearch/Dockerfile \
  && docker-push-hub apps/jupyter/Dockerfile.buildx \
  && docker-push-hub apps/demo-ui/Dockerfile.buildx \
  && docker-push-hub apps/remote-processor-service/Dockerfile.buildx \
  && return 0
  return 1
}

runtests() {
  docker system prune -f --volumes
  docker compose up reset
  poetry run pytest apps/integration/ -p integration.conftest --noconftest --docker-tag "${TAG}"
  # this is a complicated command, so:
  # -p integration.conftest - load conftest with plugins, to capture the custom command line arg (--docker-tag)
  # --noconftest            - don't load conftest at pytest runtime; it's already loaded
  # --docker-tag            - specify tag of containers to test
  return $?
}

docker-build-hub() {
  local docker_file="$1"
  [[ -n "${docker_file}" ]] || { error "missing ${docker_file}" && return 1;}
  local repo_name="$(_docker-repo-name "${docker_file}")"
  [[ -n "${repo_name}" ]] || { error "empty repo name" && return 1;}
  shift

  echo
  echo "Building in sycamore and pushing to docker hub with repo name '${repo_name}'"
  docker buildx build "$(_docker-build-args)" -t "${repo_name}:${TAG}" -f "${docker_file}" \
     --cache-to type=registry,ref="${repo_name}:build-cache",mode=max \
     --cache-from type=registry,ref="${repo_name}:build-cache" \
     "$@" --load . || { error "buildx failed" && return 1;}
  echo "Successfully built using docker file $docker_file"
}

docker-push-hub() {
  local docker_file="$1"
  [[ -n "${docker_file}" ]] || { error "missing ${docker_file}" && return 1;}
  local repo_name="$(_docker-repo-name "${docker_file}")"
  [[ -n "${repo_name}" ]] || { error "empty repo name" && return 1;}

  echo
  echo "Pushing image to docker hub for repo '${repo_name}"
  docker push "${repo_name}:${TAG}" || { error "docker push failed" && return 1;}
  echo "Successfully pushed image previously built from dockerfile ${docker_file}"
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
  local branch="$(git status | head -n1 | grep -i 'On branch ' | awk '{print $3}')"
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
