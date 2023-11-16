# quickstart

You can easily get started with the Aryn Search Platform locally using Docker. This is our recommended way to experiment with the platform. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). 

The quickstart will deploy the Aryn platform, consisting of four containers: Sycamore importer, Sycamore HTTP crawler, Aryn OpenSearch, and Aryn demo conversational search UI. Our images currently support XXXX hardware, and we tag each image with XXXXXX. 

The quickstart configuration of Aryn automactially runs an example workload that crawls XXXXX and loads it into the Aryn platform. This gives you a out-of-the-box example for conversational search. Please visit XXXXhere for instructions on how to load your own data.

The quickstart requires:

- An OpenAI Key for LLM access
- AWS credentials for Amazon Textract. The demo Sycamore processing job uses Textract for table extraction. You will accrue AWS charges for Textract usage.
- What else??

To run the quickstart:

```
cd path/to/quickstart/docker_compose
Set OPENAI_API_KEY, e.g. export OPENAI_API_KEY=sk-...XXXXX
docker compose up
```

You can then visit http://localhost:3000/ for conversational search on the sample dataset.

### Add your own data

You can create additional indexes in the quickstart with your own data and have conversational search on it. The quickstart includes a sample Sycamore script that loads and prepares an example dataset from www.sortbenchmark.org. You can also have Aryn use this script to process your own data by configuring the quickstart to ingest it:

STEPS TO DO THIS

Please note that the sample Sycamore script was created to process the data found in the Sort Benchmark dataset, and not optimized for preparing your private data for search. We recommend iterating on the Sycamore script to find the best way to prepare and enrich your data for the best quality results.

### Improve search quality on your data

Enterprise data is diverse, and Sycamore makes it easy to prepare your data for high-quality search responses. Do do this, you will likely need to have a few iterations on your Sycamore processing script, and create several indexes to test the quality of your search results. We recommend two options for this process:

**1. Iterate in the Sycamore Importer container:**
You can edit or supply a new Sycamore script to process your data in the Sycamore Importer container. You can install a text editor in the container, and then edit the script:

```
XXXXXX
```

In the script, you specifcy the name of the index to load the data into. If it doesn't already exist, the new index is created. The demo UI makes it easy to select the index you want to for conversational search.

To run the script:

```
XXXXX
````

**2. Iterate with a local version of Sycamore:**
You may prefer to use a local IDE or notebook to iterate on your Sycamore script. You can install Scyamore locally, and configure it to load the output to the Aryn OpenSearch container from the quickstart.

To install Syscamore:

```
pip install sycamore-ai
```

For certain PDF processing operations, you also need to install poppler, which you can do with the OS-native package manager of your choice. For example, the command for Homebrew on Mac OS is brew install poppler.

To conifgure Sycamore to load the local Aryn OpenSearch container:

```
# Write the embedded documents to a local OpenSearch index.
os_client_args = {
    "hosts" : [{"host": "localhost", "port": 9200}],
    "use_ssl" : True,
    "verify_certs" : False,
    "http_auth" : ("admin", "admin")
}
```





