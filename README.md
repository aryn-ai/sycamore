# Remote Processor Service
Service and library for remote processors

## Instructions 
Setting this up takes a couple steps. First, get the protocols submodule with
```
git submodule update --init --remote
```

Also install the poetry packages and stuff
```
poetry install --no-root
```

Next, generate the grpc/protobuf code. Due to some weirdness in the way protobuf python handles imports I wrote a script that screws with the directory structure (only for the grpc generate call)
```
./genrpc
```

Now, assemble a zip for the opensearch plugin by following the directions in [that repo](https://github.com/aryn-ai/opensearch-remote-processor). Copy the resulting zip into `docker/` and then build an opensearch image
```
cp ../opensearch-remote-processor/build/distributions/remote-processor-2.12.0-SNAPSHOT.zip docker
docker build -t rps-os -f docker/Dockerfile.os docker
```

Also build a docker image for the remote processor service itself
```
docker build -t rps .
```

Finally, docker compose up
```
docker compose -f docker/compse.yml up
```
Alternately, just start the OpenSearch container and run RPS locally:
```
docker run -it --rm --network=host -e discovery.type=single-node rps-os
poetry run server configs/cfg1.yml
```

And you should have an opensearch with the remote processor plugin installed and a remote processor service (running the config at `configs/cfg1.yml` - just the debug processor atm)

Now, to create a remote processor
```
curl -X PUT http://localhost:9200/_search/pipeline/remote_pipeline --json '
{
    "response_processors": [
        {
            "remote_processor": {
                "endpoint": "rps:2796/RemoteProcessorService/ProcessResponse",
                "processor_name": "debug"
            }
        }
    ]
}'
```

Test the processor server in pure python with:
```
from gen.response_processor_service_pb2_grpc import RemoteProcessorServiceStub
import grpc
from gen.response_processor_service_pb2 import ProcessResponseRequest

chan = grpc.insecure_channel('localhost:2796')
stub = RemoteProcessorServiceStub(chan)
req = ProcessResponseRequest()
req.processor_name = "debug"
res = stub.ProcessResponse(req)
print(res)
```
```
search_response {
}
```