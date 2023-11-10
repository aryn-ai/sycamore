# Doesn't work with 3.12
# depends on pyarrow==12.0.1 and ray[default]<3.0.0 and >=2.7.0
FROM python:3.11

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY importer/docker-apt-packages.sh .
RUN /bin/bash /app/docker-apt-packages.sh

COPY pyproject.toml poetry.lock README.md importer/docker-poetry-packages.sh ./
RUN /bin/bash /app/docker-poetry-packages.sh

COPY importer ./importer
COPY sycamore ./sycamore
COPY examples/simple_config.py ./importer

RUN poetry install --only-root && rm -rf $POETRY_CACHE_DIR

CMD [ "poetry", "run", "python", "importer/docker_local_import.py", "/app/.scrapy" ]
