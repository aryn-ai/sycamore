help:
	@echo "install_poetry: installs poetry on ubuntu"
	@echo "clean: clean up grpc-generated code (by deletion)"
	@echo "build_proto: generate code from the .proto files in protocols/proto-remote-processor"
	@echo "install_rps: installs dependencies, builds proto, then installs the package"
	@echo "common_build: install main dependencies"
	@echo "server_build: install the package, but not the dependencies"

install_poetry:
	apt update
	DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends python3-poetry gcc python3-dev
	apt clean
	-rm -rf /var/lib/apt/lists/*
	poetry config virtualenvs.path /rps/poetry_venv

clean:
	-rm -rd proto_remote_processor

build_proto:
	poetry install --no-root --with build
	poetry run python -m grpc_tools.protoc -I protocols/ --python_out=. --pyi_out=. --grpc_python_out=. protocols/proto-remote-processor/*.proto

install_rps:
	poetry install --no-root
	make build_proto
	poetry install --only-root

common_build:
	poetry install --no-root --only main

server_build:
	poetry install --only-root

aryn_user:
	groupadd --gid 1000 aryn
	useradd --uid 1000 --gid 1000 --home-dir /aryn --password=y --no-create-home aryn
	chown -R aryn:aryn /aryn
