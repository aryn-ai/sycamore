# Doesn't work with 3.12
# depends on pyarrow==12.0.1 and ray[default]<3.0.0 and >=2.7.0
FROM python:3.11

WORKDIR /app
COPY docker-install-packages.sh .
RUN ls /app
RUN /bin/sh /app/docker-install-packages.sh
RUN apt install -y poppler-utils
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install -vvv --no-root
COPY . .

CMD [ "poetry", "run", "python", "examples/docker_local_ingest.py", "/app/.scrapy" ]
