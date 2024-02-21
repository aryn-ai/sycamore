
ARG POETRY_NO_INTERACTION=1
ARG POETRY_VIRTUALENVS_IN_PROJECT=1
ARG POETRY_VIRTUALENVS_CREATE=1 \
ARG POETRY_CACHE_DIR=/tmp/poetry_cache
ARG RPS_PORT=2796

##########
# Common: resolve dependencies
FROM python:3.11 AS rps_common

# Is there some way to keep this common layer common across all our services?
# E.g. maybe we can have an image called 'aryn_service_base' or something
# - setup aryn user and directory
# - install commonly used software (poetry, maybe protoc)
# And then we can just layer services on top?
WORKDIR /aryn
COPY ./Makefile ./
RUN make aryn_user
RUN make install_poetry

USER aryn
WORKDIR /aryn/rps/
COPY --chown=aryn:aryn ./poetry.lock ./pyproject.toml ./
RUN make -f ../Makefile common_build

##########
# Build: build package, compile protobufs
FROM rps_common AS rps_build

# Build the proto files into python
COPY --chown=aryn:aryn ./protocols ./protocols
RUN make -f ../Makefile build_proto

##########
# Run: run the server
FROM rps_common AS rps_server

COPY --from=rps_build --chown=aryn:aryn /aryn/rps/proto_remote_processor ./proto_remote_processor
COPY --chown=aryn:aryn ./lib ./lib/
COPY --chown=aryn:aryn ./service ./service
COPY --chown=aryn:aryn ./README.md ./README.md
COPY --chown=aryn:aryn ./configs ./configs
RUN make -f ../Makefile server_build

EXPOSE $RPS_PORT

CMD ["poetry", "run", "server", "configs/cfg1.yml"]
