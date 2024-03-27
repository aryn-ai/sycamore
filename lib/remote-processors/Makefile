help:
	@echo "install_rps: installs dependencies, builds proto, then installs the package"
	@echo "clean: clean up grpc-generated code (by deletion)"
	@echo "build_proto: generate code from the .proto files in protocols/proto-remote-processor"

clean:
	-rm remote_processors/*pb2*

build_proto:
	poetry run python -m grpc_tools.protoc -I opensearch-remote-processor/src/main/proto --python_out=remote_processors --pyi_out=remote_processors --grpc_python_out=remote_processors opensearch-remote-processor/src/main/proto/*.proto
	# Fix the relative imports
	poetry run protol --in-place --python-out remote_processors protoc --proto-path=opensearch-remote-processor/src/main/proto opensearch-remote-processor/src/main/proto/*.proto

install_rps:
	poetry install --no-root
	make build_proto
	poetry install --only-root
