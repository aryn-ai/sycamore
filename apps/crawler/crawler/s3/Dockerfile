# Repo name: arynai/sycamore-crawler-s3

# In the root directory:
# docker build -t sycamore_crawler_s3 -f crawler/s3/Dockerfile
# docker run -it -v crawl_data:/app/.data/.s3 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN sycamore_crawler_s3

# Match Python version UI uses.  PyTorch not compatible with 3.12 yet.
FROM python:3.11

WORKDIR /app
COPY docker-app-user.sh .
RUN /bin/bash /app/docker-app-user.sh

RUN pip install poetry

USER app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY apps/crawler/pyproject.toml apps/crawler/poetry.lock apps/crawler/README.md ./
RUN poetry install --only crawler_s3 --no-root && rm -rf $POETRY_CACHE_DIR

COPY apps/crawler/crawler/s3 .
# Hack because pyproject.toml expects a sycamore directory
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

ENTRYPOINT [ "poetry", "run", "python", "./crawler/s3_crawler.py"]
