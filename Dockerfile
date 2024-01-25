FROM mwalbeck/python-poetry:1.7-3.11

WORKDIR /rps
COPY . /rps/
RUN poetry install
EXPOSE 2796
CMD ["poetry", "run", "server", "configs/cfg1.yml"]
