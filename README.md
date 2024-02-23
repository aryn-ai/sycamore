# Remote Processor Service
This project aims to enable "Remote Search Processors". OpenSearch allows post-processing of search responses and requests 
via an object called a "Search Processor". We've found that implementing these in java and building and installing a plugin
every time we want to change the behavior of such a search processor produces a very long iteration cycle. Our solution is 
to simply pull the search processors out of OpenSearch entirely; instead we install a single plugin that makes RPCs to an 
external service which contains the processors. This is the external service; the plugin is [over here](https://github.com/aryn-ai/opensearch-remote-processor).

In order to keep as much parity with the OpenSearch Search Processor interface, the plugin and external service communicate
through protocol-buffered forms of the OpenSearch-internal SearchResponse and SearchRequest objects. The protobuf definitions
can be found [over here](https://github.com/aryn-ai/protocols).

We have a little bit of a nomenclature clash. There are two levels of processors in the service: processors and pipelines. 
A processor is a single unit of processing; whereas a pipeline is made up of a string of processors. You can build a 
one-processor pipeline. The trick is that the service exposes each pipeline as an RPC to OpenSearch, and in OpenSearch the
Remote Search Processor calls out to one of these pipelines. So you will have an OpenSearch search pipeline with a remote
search processor that calls out to a remote search pipeline endpoint made up of search processors. Try not to think about
it too much. Instead, look at the picture:

![untitled](img/RPS_Architecture.svg)

## Instructions 
Setting this up takes a couple steps. First, get the protocols submodule with
```
git submodule update --init --remote
```

Also install the poetry packages and stuff
```
poetry install --no-root
```

Next, generate the grpc/protobuf code. Due to some weirdness in the way protobuf python handles imports I wrote a script that screws with the directory structure (only for the grpc generate call).
Once the grpc code is generated you can install the package itself.
```
make build_proto
poetry install
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
docker compose -f docker/compose.yml up
```
Alternately, one can start the OpenSearch container and run RPS locally.
Be sure to change `rps` to `localhost` in the endpoint below.
```
docker run -d --rm --network=host -e discovery.type=single-node rps-os
poetry run server config/pipelines.yml
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

Or, test the processor via OpenSearch:
```
curl 'http://localhost:9200/demoindex0/_search?search_pipeline=remote_pipeline&pretty' --json '
{
  "query": {
    "match": {
      "text_representation": "armadillo"
    }
  }
}'
```
