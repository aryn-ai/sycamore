FROM nikolaik/python-nodejs:python3.11-nodejs20

WORKDIR /home/pn/js-ui
COPY ui/package.json ui/package-lock.json ./
RUN npm install && npm cache clean --force

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
CMD /bin/bash run-ui.sh
