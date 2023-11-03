FROM python:3

WORKDIR /app
RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR
COPY http .
RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR
ENTRYPOINT [ "./run-crawler.sh" ]
