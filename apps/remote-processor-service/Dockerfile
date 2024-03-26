# Repo name: arynai/sycamore-remote-processor-service

##########
# Common: resolve dependencies
FROM python:3.11 AS rps_common

ARG RPS_PORT=2796
ARG POETRY_NO_INTERACTION=1
ARG POETRY_VIRTUALENVS_IN_PROJECT=1
ARG POETRY_VIRTUALENVS_CREATE=1
ARG POETRY_CACHE_DIR=/tmp/poetry_cache

# Is there some way to keep this common layer common across all our services?
# E.g. maybe we can have an image called 'aryn_service_base' or something
# - setup aryn user and directory
# - install commonly used software (poetry, maybe protoc)
# And then we can just layer services on top?

WORKDIR /aryn/
COPY ./Makefile ./
RUN make aryn_user
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    make install_poetry

USER aryn
WORKDIR /aryn/rps/
COPY --chown=aryn:aryn ./poetry.lock ./pyproject.toml ./
RUN --mount=type=cache,id=cache_poetry_1000,target=/tmp/poetry_cache,uid=1000,gid=1000,sharing=locked \
    make -f ../Makefile common_build

##########
# Build: build package, compile protobufs
FROM rps_common AS rps_build

# Build the proto files into python
COPY --chown=aryn:aryn ./opensearch-remote-processor ./opensearch-remote-processor
COPY --chown=aryn:aryn ./lib ./lib
RUN --mount=type=cache,id=cache_poetry_1000,target=/tmp/poetry_cache,uid=1000,gid=1000,sharing=locked \
    make -f ../Makefile docker_build_proto

##########
# Run: run the server
FROM rps_common AS rps_server

COPY --from=rps_build --chown=aryn:aryn /aryn/rps/lib ./lib
COPY --chown=aryn:aryn ./service ./service
COPY --chown=aryn:aryn ./README.md ./README.md
COPY --chown=aryn:aryn ./config ./config
COPY --chown=aryn:aryn ./rps_docker_entrypoint.sh ./
RUN make -f ../Makefile server_build
RUN chmod +x rps_docker_entrypoint.sh
RUN make -f ../Makefile user_check

EXPOSE $RPS_PORT

CMD ./rps_docker_entrypoint.sh
