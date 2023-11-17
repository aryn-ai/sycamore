# Quickstart for Aryn Search - Docker compose

You can easily get started with the Aryn Search Platform locally using Docker. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). 

The quickstart will deploy the Aryn platform, consisting of four containers: Sycamore importer, Sycamore HTTP crawler, Aryn OpenSearch, and Aryn demo conversational search UI. Our images currently support linux/amd64 and linux/arm64 hardware.

The quickstart configuration of Aryn automatically runs an example workload that crawls the [Sort Benchmark website](http://www.sortbenchmark.org), downloads [this PDF](http://sortbenchmark.org/2004_Nsort_Minutesort.pdf), and loads it into the Aryn platform. This gives you a out-of-the-box example for conversational search. Please [see below](###add-your-own-data) for instructions on how to load your own data.

## Deploying Aryn Search

### Quickstart prerequisites

1. An OpenAI Key for LLM access. You can create an OpenAI account [here](https://platform.openai.com/signup), or if you already have one, you can retrieve your key [here](https://platform.openai.com/account/api-keys)

2. For the highest quality answers, AWS credentials for Amazon Textract and an Amazon S3 bucket for Textract input/output. The demo Sycamore processing job uses Textract for table extraction. You will accrue AWS charges for Textract usage. When setting up your environment below, you can choose to disable Textract access for simplicity, but the processing and answer quality will be lower quality.

For using Textract:

- If you do not have an AWS account, sign up [here](https://portal.aws.amazon.com/billing/signup).  
- Create an Amazon S3 bucket in that account for use with Textract (e.g.  e.g. s3://username-textract-bucket). We recommend you set up bucket lifecycle rules that automatically delete files in this bucket, as the data stored here is only needed temporarily during a Sycamore data processing job.  

### Now, let's get started  

1. Download the Docker compose files for the Quickstart [here](https://github.com/aryn-ai/quickstart/tree/main/docker_compose)  

2. Set up your Aryn Search environment:

```
export SYCAMORE_TEXTRACT_PREFIX=s3://your-bucket-name-here
export OPENAI_API_KEY=YOUR-KEY
```

Textract is used by default with the demo Sycamore script, and you can choose to enable it or disable it. 

- To use it, you need to configure your AWS credentials. You can enable AWS SSO login with [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-token-auto-sso), or you can use other methods to set up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and if needed AWS_SESSION_TOKEN. 

```
aws sso login --profile YOUR-PROFILE-NAME
eval "$(aws configure export-credentials --format env --profile YOUR-PROFILE-NAME)"
```

You can verify it is working by running:

```
aws s3 ls --profile YOUR-PROFILE-NAME
```
You should see the bucket you created for $SYCAMORE_TEXTRACT_PREFIX.  

- To disable Textract in your Sycamore job, then you can turn it off by:

```
export ENABLE_TEXTRACT=false
```

3. Start the Docker service  
   a. On MacOS or Windows, start Docker desktop  
   b. On Linux, if you used your local package manager, it should be started  

5. Adjust Docker service memory settings  
In Docker, go to "Settings" - on MacOS, it's the gear icon in the top right of the UI. Next, click on "Resources" and adjust Memory limit to 6 GB and Swap to 4 GB. If you are seeing memory issues while running Aryn Search, you can add adjust memory allocation here.

6. Start Aryn Search
In the directory where you downloaded the Docker compose files, run:

```
docker compose up 
```

Once you see log messages similar to:

```
No changes at [datetime] sleeping
```

Then Aryn Search has properly processed the PDF and loaded it into the index.

7. Use the demo UI for conversational search

- Using your internet browser, visit http://localhost:3000.
- Create a new conversation. Enter the name for your conversation in the text box in the left "Conversations" panel, and hit enter or click the "add convo" icon on the right of the text box.
- Select your conversation, and then write a question into the text box in the middle panel. Hit enter.
- Ask follow up questions. You'll see the actual results from the Aryn Search hybrid search for your question in the right panel, and the conversational search in the middle panel.  

Congrats! You've deployed Aryn Search and enabled conversational search over a document. Next, you can choose to ingest the rest of the documents from the [Sort Benchmark website](##add-sort-benchmark-dataset) or your own data(##add-your-own-data). 

## Add Sort Benchmark Dataset

By default, the Quickstart crawls the [Sort Benchmark website](http://www.sortbenchmark.org), downloads [this PDF](http://sortbenchmark.org/2004_Nsort_Minutesort.pdf). However, you may want to ingest the whole Sort Benchmark website dataset to search over more documents. This dataset includes many PDFs and the acutal HTML pages themselves, and has a variety of tables (some very poorly formatted!) and figures. After loading this data, you can experiment with how Aryn Search can answer questions on this unstructured dataset.

Keep the Aryn Stack running from the previous example, and then:

1. Run the Sycamore HTTP Crawler container with an additional parameter:
```
docker compose -f sort-all.yaml up
```
This will crawl and download the data from the Sort Benchmark website. 

2. Sycamore will automatically start processing the new data. Once you see log messages similar to:

```
No changes at [datetime] sleeping
```

Then Aryn Search has properly processed the data and loaded it into the index. You can interact with the UI while it is loading, but the data won't all be available.


## Add your own data

You can create additional indexes in the quickstart with your own data and have conversational search on it. The quickstart includes a sample Sycamore script that loads and prepares an example dataset from https://sortbenchmark.org. You can also have Aryn use this script to process your own data by configuring the quickstart to ingest it:

STEPS TO DO THIS

Please note that the sample Sycamore script was created to process the data found in the Sort Benchmark dataset and not optimized for preparing your private data for search. We recommend iterating on the Sycamore script to find the best way to prepare and enrich your data for the best quality results.

## Improve search quality on your data

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

To conifgure Sycamore to ingest into the local Aryn OpenSearch container:

```
# Write the embedded documents to a local OpenSearch index.
os_client_args = {
    "hosts" : [{"host": "localhost", "port": 9200}],
    "use_ssl" : True,
    "verify_certs" : False,
    "http_auth" : ("admin", "admin")
}
```

## Troubleshooting

### Nothing is importing

Look at the log messages for the sycamore container: `docker compose logs sycamore`

1. If you are missing environment variables, follow the instructions above to set them up.

1. If it is constantly OOMing, increase the memory for your docker container
   1. Via Docker Desktop: Settings > Resources > scroll down > ...
   1. The minimum successfully tested resources were 4GB RAM, 4GB Swap, 64GB disk space

### Reset the configuration entirely

If the persistent data is in a confused state, you can reset it back to a blank state:

`% docker compose -f reset.yaml up`

### Reach out for help on the sycamore slack channel

https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg
