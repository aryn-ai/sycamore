help:
	@echo "install_rps: installs dependencies, builds proto, then installs the package"
	@echo "clean: clean up grpc-generated code (by deletion)"
	@echo "build_proto: generate code from the .proto files in protocols/proto-remote-processor"

clean:
	-rm lib/*pb2*

build_proto:
	poetry run python -m grpc_tools.protoc -I opensearch-remote-processor/src/main/proto --python_out=lib --pyi_out=lib --grpc_python_out=lib opensearch-remote-processor/src/main/proto/*.proto
	# Fix the relative imports
	poetry run protol --in-place --python-out lib protoc --proto-path=opensearch-remote-processor/src/main/proto opensearch-remote-processor/src/main/proto/*.proto

install_rps:
	poetry install --no-root
	make build_proto
	poetry install --only-root

### Docker steps sorted in the same order as the Dockerfile
### Not for general use so undocumented

aryn_user:
	groupadd --gid 1000 aryn
	useradd --uid 1000 --gid 1000 --home-dir /aryn --password=y --no-create-home aryn
	chown -R aryn:aryn /aryn

install_poetry:
	touch /var/lib/apt/.cache_var_lib_apt # make it possible to find the cache directory for buildx builds
	touch /var/cache/apt/.cache_var_cache_apt
	apt update
	DEBIAN_FRONTEND=noninteractive apt -y install --no-install-recommends python3-poetry gcc python3-dev

common_build:
	test "$(POETRY_CACHE_DIR)" = /tmp/poetry_cache # catch a bug where putting ARG too early in Dockerfile doesn't get the env var
	touch /tmp/poetry_cache/.poetry_cache_dir
	poetry install --no-root --only main
	poetry env info

docker_build_proto:
	test "$(POETRY_CACHE_DIR)" = /tmp/poetry_cache
	poetry install --no-root --only build
	make -f ../Makefile build_proto

server_build:
	poetry install --only-root

user_check:
	FILES=$$(find /aryn -print | wc -l); \
	ARYN_FILES=$$(find /aryn -user aryn -print | wc -l); \
	echo "Found $${ARYN_FILES}/$${FILES} owned by aryn"; \
	find /aryn ! -user aryn -print; \
	test $${FILES} -ge 1000 && \
	test $${ARYN_FILES} -eq $${FILES}
