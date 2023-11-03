# Doesn't work with 3.12
# depends on pyarrow==12.0.1 and ray[default]<3.0.0 and >=2.7.0
FROM python:3.11

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY docker-install-packages.sh .
RUN /bin/sh /app/docker-install-packages.sh
COPY examples/docker-preload-models.pdf docker-download-data.sh ./
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,docker --no-root -v && rm -rf $POETRY_CACHE_DIR
COPY . .
RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR
RUN /bin/sh /app/docker-download-data.sh

CMD [ "poetry", "run", "python", "examples/docker_local_ingest.py", "/app/.scrapy" ]
