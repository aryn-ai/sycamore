# Repo name: arynai/sycamore-importer

# Doesn't work with 3.12
# depends on pyarrow==12.0.1 and ray[default]<3.0.0 and >=2.7.0
FROM python:3.11

WORKDIR /app

ARG POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY apps/importer/importer/docker/docker-apt-packages.sh .
RUN /bin/bash /app/docker-apt-packages.sh

COPY apps/importer/pyproject.toml apps/importer/poetry.lock apps/importer/README.md \
    apps/importer/importer/docker/docker-poetry-packages.sh ./apps/importer/
COPY apps/importer apps/importer
COPY lib/sycamore lib/sycamore

WORKDIR /app/apps/importer
RUN /bin/bash /app/apps/importer/docker-poetry-packages.sh

COPY examples/simple_config.py ./

RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR

ARG GIT_BRANCH="main"
ARG GIT_COMMIT="unknown"
ARG GIT_DIFF="unknown"

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}

CMD [ "poetry", "run", "python", "importer/docker/docker_local_import.py", "/app/.scrapy" ]
