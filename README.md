# quickstart

You can easily get started with the Aryn Search Platform locally using Docker. This is our recommended way to experiment with the platform. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). 

The quickstart will deploy the Aryn platform, consisting of four containers: Sycamore importer, Sycamore HTTP crawler, Aryn OpenSearch, and Aryn demo conversational search UI. Our images currently support XXXX hardware, and we tag each image with XXXXXX. 

The quickstart configuration of Aryn automactially runs an example workload that crawls XXXXX and loads it into the Aryn platform. This gives you a out-of-the-box example for conversational search. Please visit XXXXhere for instructions on how to load your own data.

The quickstart requires:

- An OpenAI Key for LLM access
- AWS credentials for Amazon Textract. The demo Sycamore job uses Textract for table extraction. You will accrue AWS charges for Textract usage.
- What else??

To run the quickstart:

```
cd path/to/quickstart/docker_compose
Set OPENAI_API_KEY, e.g. export OPENAI_API_KEY=sk-...XXXXX
docker compose up
```

You can then visit http://localhost:3000/ for conversational search on the sample dataset.

### Add your own data with demo Sycamore script

You can create additional indexes in the quickstart with your own data and have conversational search on it. The quickstart includes a sample Sycamore script that loads and prepares an example dataset from www.sortbenchmark.org. You can also have Aryn use this script to process your own data by configuring the quickstart to ingest it:

STEPS TO DO THIS

Please note that the sample Sycamore script was created to process the data found in the Sort Benchmark dataset, and not optimized for preparing your private data for search. We recommend iterating on the Sycamore script to find the best way to prepare and enrich your data for the best quality results.

### Add your own data with demo Sycamore script



