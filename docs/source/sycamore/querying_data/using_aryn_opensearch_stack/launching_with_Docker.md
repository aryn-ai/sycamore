# Using Docker

You can also deploy Sycamore using Docker, and you can launch it locally or on a virtual machine. This is a legacy way to run Sycamore, and we recommend installing the libary directly. The Docker compose launches a stack including a Jupyter notebook, OpenSearch hybrid search engine, and natural language search UI with retrieval-augmented generation (RAG). 

If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). The Sycamore Docker images currently support amd64 and arm64 hardware.

1. Clone the Sycamore repo:

```bash
git clone https://github.com/aryn-ai/sycamore
```

2. Create OpenAI Key for LLM access. Sycamore’s default configuration uses OpenAI for RAG and entity extraction. You can create an OpenAI account [here](https://platform.openai.com/signup), or if you already have one, you can retrieve your key [here](https://platform.openai.com/account/api-keys).

3. Set OpenAI Key:

```bash
export OPENAI_API_KEY=YOUR-KEY
```

4. Go to `/sycamore`

5. Launch Sycamore. Containers will be pulled from DockerHub:

```bash
docker compose up --pull=always
```

Note: You can alternately remove the `--pull=always` and instead run `docker compose pull` to control when new images are downloaded. `--pull=always` guarantees you have the most recent images for the specified version.


>By default, `docker compose` uses the stable version of the containers. You can choose a specific version to run, e.g. `latest` (last build pushed), `latest_rc` (last release candidate), or `0.YYYY.MM.DD` (date-stamped release). To specify a version, set the `VERSION` environment variable, e.g. `VERSION=latest_rc docker compose up --pull=always`. See the `.env` file if you want to specify different versions for the separate containers.


## Optional: Configure HTTPS Encryption

If you want to run the demo query UI and the Jupyter UI over HTTPS using SSL/TLS encryption, do the following after step 3 in the Launch Sycamore section.

`export SSL=1`

This will create self-signed certificates for `localhost` and will likely trigger warnings in browsers.  Look [here](encryption.md) for more details.


## Use a Jupyter notebook

A [Jupyter](https://jupyter.org/) notebook is a development environment that lets you
write and experiment with different data preparation jobs to improve the results of
processing your documents. The [using Jupyter
tutorial](../tutorials/sycamore-jupyter-dev-example.md) will walk you through using the containerized Jupyter notebook included with Sycamore or installing it locally, and preparing a new set of documents.


## Clean Up

To clean up your Sycamore resources:

1. Stop the containers and run the reset script:

```
docker compose down
docker compose run reset
```

# Architecture
The Sycamore Docker deployment offers a one-stop shop for search and analytics on complex unstructured data, and has several high-level components. You can also view an [architecture diagram below.](#architecture-diagram)

## Data Ingestion and Preparation

* **Crawlers:** These containers take data from a specified location (e.g. Amazon S3 bucket or website) and store it in files that can be processed by the Importer. The crawlers will only download updated or new data.

* **Importer:** This container runs data preparation workloads with operations such as data cleaning, information extraction, enrichment, summarization, and the generation of vector embeddings that encapsulate the semantics of data. It will then load prepared data into Sycamore’s vector and keyword indexes. Sycamore enables Generative AI User Defined Functions (UDFs) with a variety of LLMs and can use various vector embedding models. It runs on Ray, an open-source framework for scaling Python workloads. The importer is responsible for error handling to minimize OOMs and avoid files that fail to import with a particular Sycamore script.

## Data Storage and Retrieval

Sycamore is built with OpenSearch, an open-source enterprise-scale search and analytics engine. It uses OpenSearch’s vector database and term-based index, hybrid (vector + keyword) search, analytical functions, and conversational memory.

* **Indexes:** Sycamore stores data in both vector and keyword-based indexes.

* **Hybrid Search (Query Execution):** This runs a vector and keyword search using the respective indexes, and then returns a single set of search results with the proper weighting from each. This leverages OpenSearch’s Neural Search, which utilizes FAISS.

* **Analytics Functions (Query Execution):** OpenSearch has a variety of analytics functions, such as group by or aggregation, that can be used in queries.

* **Conversational Memory:** This is used to store each interaction in a conversation and is stored in an OpenSearch index. Applications like conversational search can utilize this as the context in a series of interactions (conversation).


## Query Layer and Post-Processing

Sycamore uses LLMs to rewrite natural language queries and implement RAG. You can also query Sycamore directly with an OpenSearch client.

* **Query Rewriting:** Currently, this functionality is only available when submitting queries through the “Demo UI” container. Sycamore will use a generative AI model to rewrite queries based on conversation history and other heuristics.

* **Retrieval-Augmented Generation (RAG):** Sycamore can execute queries using a RAG pipeline in combination with a specified LLM. It can be used directly with an OpenSearch client (using the RAG Search Pipeline) or through the Demo UI.

* **Post Retrieval Processing:** Sycamore runs post-processing steps. For instance, it includes a customizable reranker to provide reranking heuristics after hybrid search returns initial results.

## Architecture Diagram

![Untitled](imgs/SycamoreDiagram_Detailed.png)


# Troubleshooting

**Data is not being added to the index**
Look at the log messages for the Sycamore container: 

`docker compose logs sycamore`

1. Check to see if you are missing environment variables. If so, please set them using the instructions earlier in the guide.

2. If it is constantly OOMing, increase the memory for your Docker container. To do this via Docker Desktop: Settings > Resources > scroll down > ...

* The minimum successfully tested resources were 4GB RAM, 4GB Swap, 64GB disk space

**Reset Sycamore entirely**

If the persistent data is in a confused state, you can reset it back to a blank state:

```
docker compose down
docker compose run reset
```

**The OpenSearch container isn't starting**

Sycamore leverages OpenSearch for indexing, hybrid search, RAG pipelines, and more.

If you see an error message like: 
`opensearch to start did not return true with 300 tries`

We have seen this happen on MacOS for two reasons:

1. Out of disk space.

* Try docker system prune to free up disk space

* Increase the disk space available via Docker Desktop: Settings > Resources > scroll down > ...

2. VM in a weird state.

* Restart docker desktop

You may also want to reset the configuration entirely, although in all the cases where we have seen these problems, no persistent state existed to be reset.

Reach out for help via email or on the Sycamore Slack channel