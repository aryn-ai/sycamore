#!/bin/zsh

poetry run python -m grpc_tools.protoc -Iprotocols/remote-processor --python_out=gen --pyi_out=gen --grpc_python_out=gen protocols/remote-processor/*
