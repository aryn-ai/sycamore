# Repo name: arynai/sycamore-crawler-http

# Match Python version UI uses.  PyTorch not compatible with 3.12 yet.
FROM python:3.11

WORKDIR /app
RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock crawler/README.md ./
RUN poetry install --only crawler_http --no-root && rm -rf $POETRY_CACHE_DIR

COPY crawler/http .
# Hack beccause pyproject.toml expects a sycamore directory
RUN mkdir sycamore && touch sycamore/__init__.py
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

ENTRYPOINT [ "./run-crawler.sh" ]
