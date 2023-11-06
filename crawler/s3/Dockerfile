FROM python:3

WORKDIR /app

RUN pip install poetry

COPY ./pyproject.toml ./
COPY ./poetry.lock ./
RUN touch README.md
COPY ./s3 .

RUN poetry install

CMD [ "poetry", "run", "python", "./crawler/s3_crawler.py"]
