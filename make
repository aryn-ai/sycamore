#!/bin/zsh

mkdir protocols/remote-processor/gen
mv protocols/remote-processor/*.proto protocols/remote-processor/gen/
sed -i '' 's:import "search:import "gen/search:g' protocols/remote-processor/gen/response_processor_service.proto

poetry run python -m grpc_tools.protoc -I protocols/remote-processor --python_out=. --pyi_out=. --grpc_python_out=. protocols/remote-processor/gen/*.proto

sed -i '' 's:import "gen/search:import "search:g' protocols/remote-processor/gen/response_processor_service.proto
mv protocols/remote-processor/gen/* protocols/remote-processor/
rmdir protocols/remote-processor/gen
