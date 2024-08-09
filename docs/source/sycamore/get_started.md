# Get Started With Sycamore

## Install Library

We recommend installing the Sycmaore library using ```pip```:

```pip install sycamore-ai ```

Next, you can set the proper API keys for related services, like the Aryn Partitioning Service (APS) for processing PDFs ([sign-up here](https://www.aryn.ai/get-started) for free) or OpenAI to use GPT with Sycamore's LLM-based transforms.

Now, that you have installed Sycamore, you see it in action using the example Jupyter notebooks. Many of these examples load a vector database in the last step of the processing pipeline, but you can edit the notebook to write the data to a different target database or out to a file. [Visit the Sycamore GitHub](https://github.com/aryn-ai/sycamore/tree/main/notebooks) for the sample notebooks.

Here are a few good notebooks to start with:

* A [notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/tutorial.ipynb) showing a simple processing job using APS to chunk PDFs, two LLM-based entity extraction transforms, and loading an OpenSearch hybrid index (vector + keyword)
* A [notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/VisualizePartitioner.ipynb) that visually shows the bounding boxes created by the Aryn Partioning Service
* A [more advanced Sycamore pipeline](https://github.com/aryn-ai/sycamore/blob/main/notebooks/metadata-extraction.ipynb) that chunks PDFs using APS, does schema extraction and population using LLM transforms, data cleaning using Python, and loads an OpenSearch hybrid index (vector + keyword)
* A [notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/pinecone-writer.ipynb) showing how to load a Pinecone vector database. There are other example notebooks showing sample code for loading other targets [here](https://github.com/aryn-ai/sycamore/tree/main/notebooks).

## Using Docker

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


### Optional: Configure HTTPS Encryption

If you want to run the demo query UI and the Jupyter UI over HTTPS using SSL/TLS encryption, do the following after step 3 in the Launch Sycamore section.

`export SSL=1`

This will create self-signed certificates for `localhost` and will likely trigger warnings in browsers.  Look [here](encryption.md) for more details.


### Use a Jupyter notebook

A [Jupyter](https://jupyter.org/) notebook is a development environment that lets you
write and experiment with different data preparation jobs to improve the results of
processing your documents. The [using Jupyter
tutorial](../tutorials/sycamore-jupyter-dev-example.md) will walk you through using the containerized Jupyter notebook included with Sycamore or installing it locally, and preparing a new set of documents.


### Clean Up

To clean up your Sycamore resources:

1. Stop the containers and run the reset script:

```
docker compose down
docker compose run reset
```

### Troubleshooting

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

## Contact us

Feel free to reach out if you have any questions, issues, or feedback:

info@aryn.ai

Or on Slack:

https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg

It will help to have the output from docker compose logs | grep Version-Info to identify which versions of the containers you are running.
