# Quickstart for Aryn Search - Docker compose

You can easily get started with the Aryn Search Platform locally using Docker. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). 

The Quickstart will deploy the Aryn platform, consisting of four containers: Sycamore importer, Sycamore HTTP crawler, Aryn OpenSearch, and Aryn demo conversational search UI. Our images currently support amd64 and arm64 hardware.

The Quickstart configuration of Aryn automatically runs an example workload that downloads [this PDF](http://sortbenchmark.org/2004_Nsort_Minutesort.pdf) from the [Sort Benchmark website](http://www.sortbenchmark.org), uses [Sycamore](https://github.com/aryn-ai/sycamore) to prepare the data, and loads it into the Aryn platform. This gives you a out-of-the-box example for conversational search. 

You can then choose to crawl the rest of the [Sort Benchmark website](http://www.sortbenchmark.org) and load this data into your index. Instructions are [here](#Add-the-full-Sort-Benchmark-Dataset). Then, you can search over the whole dataset.

Finally, [see below](#add-your-own-data) for instructions on how to crawl and load data from any website using this Sycamore processing script.

## Deploying Aryn Search

### Quickstart prerequisites

1. An OpenAI Key for LLM access. You can create an OpenAI account [here](https://platform.openai.com/signup), or if you already have one, you can retrieve your key [here](https://platform.openai.com/account/api-keys).

2. For the highest quality table extraction (and better answers), the demo Sycamore script needs AWS credentials for Amazon Textract and an Amazon S3 bucket for Textract input/output. This script uses Textract and Amazon S3 in the US-East-1 region. You can optionally disable Textract. You will accrue AWS charges for Textract usage. If you want to enable Textract:

- If you do not have an AWS account, sign up [here](https://portal.aws.amazon.com/billing/signup). You will need this during configuration.
- Create an Amazon S3 bucket in your AWS account in the US-East-1 region for use with Textract (e.g. s3://username-textract-bucket). We recommend you set up bucket lifecycle rules that automatically delete files in this bucket, as the data stored here is only needed temporarily during a Sycamore data processing job.  

### Now, let's get started  

1. Download the Docker compose files from the Quickstart repo. You will need the [compose](https://github.com/aryn-ai/quickstart/blob/main/compose.yaml) and [.env](https://github.com/aryn-ai/quickstart/blob/main/.env) files.

```
git clone git@github.com:aryn-ai/quickstart.git
```

2. Set up your Aryn Search environment:

```
export SYCAMORE_TEXTRACT_PREFIX=s3://your-bucket-name-here
export OPENAI_API_KEY=YOUR-KEY
```

Textract is used by default with the demo Sycamore script, and you can choose to enable it or disable it. 

- To use it, you need to configure your AWS credentials. You can enable AWS SSO login with [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-token-auto-sso), or you can use other methods to set up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and if needed AWS_SESSION_TOKEN.

If using AWS SSO:

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

4. Adjust Docker service memory settings  
In Docker, go to "Settings" (e.g. on MacOS, it's the gear icon in the top right of the UI). Next, click on "Resources" and adjust Memory limit to 6 GB and Swap to 4 GB. If you are seeing memory issues while running Aryn Search, you can add adjust memory allocation here.

5. Start Aryn Search
In the directory where you downloaded the Docker compose files, run:

```
docker compose up --pull=always
```

NOTE: You can alternately remove the `--pull=always` and instead run `docker compose pull` to
control when new images are downloaded. `--pull=always` guarantees you have the most recent images
for the specified version.

Aryn Search will start up and run the demo Sycamore script, process the data, and load the index. You will know when these steps are completed when you see log messages similar to:

```
No changes at [datetime] sleeping
```

NOTE: by default the docker compose uses the stable version of the containers. You can choose a
specific version to run, e.g. latest (last build pushed), latest_rc (last release candidate), or
0.YYYY.MM.DD (date-stamped release). To specify a version set the VERSION environment variable,
e.g. `VERSION=latest_rc docker compose up --pull=always`. See the .env file if you want to specify
different versions for the separate containers.

6. Use the demo UI for conversational search

- Using your internet browser, visit http://localhost:3000.
- Create a new conversation. Enter the name for your conversation in the text box in the left "Conversations" panel, and hit enter or click the "add convo" icon on the right of the text box.
- Select your conversation, and then write a question into the text box in the middle panel. Hit enter.
- Ask follow up questions. You'll see the actual results from the Aryn Search hybrid search for your question in the right panel, and the conversational search in the middle panel.  

Congrats! You've deployed Aryn Search and enabled conversational search over a document. Next, you can choose to ingest the rest of the documents from the [Sort Benchmark website](##add-the-full-sort-benchmark-dataset) to search over more data. 

## Add the full Sort Benchmark Dataset

By default, the Quickstart downloads and ingests [this PDF](http://sortbenchmark.org/2004_Nsort_Minutesort.pdf) from the [Sort Benchmark website](http://www.sortbenchmark.org). However, you may want to ingest the whole Sort Benchmark website dataset to search over more documents. This dataset includes many PDFs and the acutal HTML pages themselves, and has a variety of tables (some very poorly formatted!) and figures. After loading this data, you can experiment with how Aryn Search can answer questions on this unstructured dataset.

Optional: Keep the Aryn Stack running from the previous example. If you shut it down, restart it using the compose command from the prior section.

You will now crawl the the Sort Benchmark website and add the rest of the data:

1. Run the Sycamore HTTP Crawler container with an additional parameter:
```
docker compose run sycamore_crawler_http_sort_all
```
This will crawl and download the data from the Sort Benchmark website. 

2. Sycamore will automatically start processing the new data. The processing job is complete and the data is loaded into the index once you see log messages similar to:

```
No changes at [datetime] sleeping
```

You can interact with the demo UI while data is being added to the index, but the data won't all be available until the job is done. Once this is complete, some sample questions to ask are:

- Who are the most recent winners of the Sort Benchmark?
- What are the hardware costs for ELSAR? Return as a table.
- What is CloudSort?


## Add a dataset from an arbitrary website

You can try using an arbitrary website with the Sort Benchmark importing script. This script is not optimized for new datasets, so the answer quality may vary on new websites. However we have found
positive results with some experiments.

WARNING: Processing data using the Sort Benchmark importing script will send your data to OpenAI,
and optionally Amazon Textract for calls to their AI services. Consider whether this is acceptable
if you are using a non-public website for testing.

1. Run the Sycamore HTTP Crawler container with an additional parameter:
```
docker compose run sycamore_crawler_http _url_
# for example
docker compose run sycamore_crawler_http http://www.aryn.ai
```

This will crawl and download the data from the specified website.  If you import aryn.ai, you can
try "who are the Aryn founders?"

2. Sycamore will automatically start processing the new data into the existing index. The processing job is complete and the data is loaded into the index once you see log messages similar to:

```
No changes at [datetime] sleeping
```

## Add a dataset from an arbitrary S3 bucket

You can try using an arbitrary pdf and html data from an S3 bucket with the Sort Benchmark
importing script. This script is not optimized for new datasets, so the answer quality may
vary. However we have found positive results with some experiments.

WARNING: Processing data using the Sort Benchmark importing script will send your data to OpenAI,
and optionally Amazon Textract for calls to their AI services. Consider whether this is acceptable
if you are using a non-public website for testing.

1. Run the Sycamore S3 Crawler container with one or two additional parameters:
```
docker compose run sycamore_crawler_s3 _bucket_ _prefix_
# for example to load the single file that's automatically downloaded via HTTP:
docker compose run sycamore_crawler_s3 aryn-public sort-benchmark/pdf/2004_Nsort
# or to load all the PDFs that are in the S3 bucket:
docker compose run sycamore_crawler_s3 aryn-public sort-benchmark/pdf
```

This will crawl and download the data from the specified S3 bucket and prefix. Note, you will need
to have the S3 authorization environment variables and be able to access the bucket.  You can check
that by running:
```
aws s3 ls _bucket_
# and to check the prefix works:
aws s3 ls _bucket_/_prefix_
```

2. Sycamore will automatically start processing the new data into the existing index. The processing job is complete and the data is loaded into the index once you see log messages similar to:

```
No changes at [datetime] sleeping
```

## Start a Jupyter notebook

A [Jupyter](https://jupyter.org/) notebook is a local development environment that lets you
interact with Sycamore and experiment with different segmentations to improve the results of
processing your documents. The [local development
instructions](sycamore-local-development-example.md) will walk you through setting up Jupyter or
running it in a container, and analyzing a new set of documents.

## Clean up

To clean up resources created in the Quickstart

1. Stop the containers
2. Run the reset script:

```
docker compose down
docker compose run reset
```

## Troubleshooting

### Data is not being added to the index

Look at the log messages for the Sycamore container: `docker compose logs sycamore`

1. Check to see if you are missing environment variables. If so, please set them using the instructions earlier in the guide.

1. If it is constantly OOMing, increase the memory for your Docker container
   1. Via Docker Desktop: Settings > Resources > scroll down > ...
   1. The minimum successfully tested resources were 4GB RAM, 4GB Swap, 64GB disk space

### Reset the configuration entirely

If the persistent data is in a confused state, you can reset it back to a blank state:

```
docker compose down
docker compose run reset
```

### Opensearch isn't starting

If you see an error message like: `opensearch to start did not return true with 60 tries`

We have seen this happen on MacOS for two reasons:

1. Out of disk space.
   1. Try `docker system prune` to free up disk space
   1. Increase the disk space available via Docker Desktop: Settings > Resources > scroll down > ...
1. VM in a weird state.
   1. Restart docker desktop

You may also want to reset the configuration entirely, although in all the cases where we have seen
these problems, no persistent state existed to be reset.

## Reach out for help via email or on the Sycamore Slack channel

Feel free to reach out if you have any questions, issues, or feedback:

info@aryn.ai 

Or on Slack:

https://join.slack.com/t/sycamore-ulj8912/shared_invite/zt-23sv0yhgy-MywV5dkVQ~F98Aoejo48Jg

It will help to have the output from `docker compose logs | grep Version-Info` to identify which
versions of the containers you are running.
