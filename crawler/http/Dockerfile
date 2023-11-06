# In the root directory:
# docker build -t sycamore_crawler_http -f crawler/http/Dockerfile
# docker run -it sycamore_crawler_http
FROM python:3

WORKDIR /app
RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock crawler/README.md ./
RUN poetry install --only main,crawler_http --no-root && rm -rf $POETRY_CACHE_DIR

COPY crawler/http .
# Hack beccause pyproject.toml expects a sycamore directory
RUN mkdir sycamore && touch sycamore/__init__.py
RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR

ENTRYPOINT [ "./run-crawler.sh" ]
