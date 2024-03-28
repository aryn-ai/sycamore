# Compose setup

See ../docker_compose/README.md for instructions

# Manual setup

## Run once:

```
docker volume create opensearch_data
```

## Run if you change the source code:
```
cd pyscripts/docker_service/opensearch
docker build -t aryn_opensearch .
```

## Run and shutdown as you see fit:
```
docker run -it --rm --name aryn_opensearch --network aryn-app -p 9200:9200 -e OPENAI_API_KEY --volume opensearch_data:/usr/share/opensearch/data aryn/opensearch
```

You can replace -it with --detach if you want it to run in the background

## To reset the opensearch state

```
docker run --volume opensearch_data:/tmp/osd ubuntu /bin/sh -c 'rm -rf /tmp/osd/*'
```
