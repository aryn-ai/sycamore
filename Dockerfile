# Repo name: arynai/demo-ui

FROM nikolaik/python-nodejs:python3.11-nodejs20

WORKDIR /home/pn/js-ui
COPY ui/package.json ui/package-lock.json ui/npm-install.sh ui/pdf.worker.js.patch ./
RUN /bin/bash npm-install.sh

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /home/pn/py-proxy
COPY openai-proxy/pyproject.toml openai-proxy/poetry.lock openai-proxy/README.md ./
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

WORKDIR /
COPY ui /home/pn/js-ui
COPY openai-proxy /home/pn/py-proxy
COPY run-ui.sh /home/pn

WORKDIR /home/pn/py-proxy
RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR

WORKDIR /home/pn

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

CMD /bin/bash run-ui.sh
