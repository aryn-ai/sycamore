# Get Started With Sycamore

## Launch Sycamore

Sycamore is deployed using Docker, and you can launch it locally or on a virtual machine. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). The Sycamore Docker images currently support amd64 and arm64 hardware.

1. Clone the Sycamore repo:

`git clone https://github.com/aryn-ai/sycamore`

4. Create OpenAI Key for LLM access. Sycamore’s default configuration uses OpenAI for RAG and entity extraction. You can create an OpenAI account [here](https://platform.openai.com/signup), or if you already have one, you can retrieve your key [here](https://platform.openai.com/account/api-keys).

5. Set OpenAI Key:

`export OPENAI_API_KEY=YOUR-KEY`

6. Go to:

`/sycamore`

9. Launch Sycamore. Containers will be pulled from DockerHub:

`Docker compose up --pull-always`

Note: You can alternately remove the `--pull=always` and instead run docker compose pull to control when new images are downloaded. `--pull=always` guarantees you have the most recent images for the specified version. 

Congrats – you have launched Sycamore! Now, it’s time to ingest and prepare some data, and run conversational search on it. Continue on to the next section to do this with a sample dataset or a website that you specify.

For more info:

* Loading data into Sycamore
* Querying your data
* Using Jupyter notebook to customize data preparation code

*Note: By default, the Docker compose uses the stable version of the containers. You can choose a specific version to run, e.g. latest (last build pushed), latest_rc (last release candidate), or 0.YYYY.MM.DD (date-stamped release). To specify a version, set the VERSION environment variable, e.g. VERSION=latest_rc docker compose up --pull=always. See the .env file if you want to specify different versions for the separate containers.*

### Optional: Configure AWS Credentials for Amazon Textract usage

Sycamore’s default data ingestion and preparation code can optionally use Amazon Textract for table extraction, which will give higher quality answers for questions on embedded tables in documents. To enable it, Sycamore needs AWS credentials for Amazon Textract and an Amazon S3 bucket for Textract input/output. The default code will use Textract and Amazon S3 in the US-East-1 region, and you will accrue AWS charges for Textract usage.

1. If you do not have an AWS account, sign up [here](https://portal.aws.amazon.com/billing/signup). You will need this during configuration.

2. Create an Amazon S3 bucket in your AWS account in the US-East-1 region for use with Textract (e.g. s3://username-textract-bucket). We recommend you set up bucket lifecycle rules that automatically delete files in this bucket, as the data stored here is only needed temporarily during a Sycamore data processing job.

3. Enable Sycamore to use Textract by setting the S3 prefix/bucket name for Textract to use:

`export SYCAMORE_TEXTRACT_PREFIX=s3://your-bucket-name-here`

4. Configure your AWS credentials. You can enable AWS SSO login with [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-token-auto-sso), or you can use other methods to set up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and if needed AWS_SESSION_TOKEN.

If using AWS SSO:

```
aws sso login --profile YOUR-PROFILE-NAME
eval "$(aws configure export-credentials --format env --profile YOUR-PROFILE-NAME)"
```

You can verify it is working by running:

`aws s3 ls --profile YOUR-PROFILE-NAME`

You should see the bucket you created for $SYCAMORE_TEXTRACT_PREFIX.


## Demo: Ingest and Query Sort Benchmark dataset

You can next run a sample workload that ingests data from the [Sort Benchmark website](http://www.sortbenchmark.org/) into Sycamore and makes it available to query through the demo query UI. This dataset contains HTML webpages and journal articles in PDFs, including images and tables. After loading and preparing this data, you can experiment with how Sycamore can answer questions on this unstructured dataset.

To answer these questions, you will need to process the data using Sycamore’s table extraction feature set. Amazon Textract support needs to be enabled, and you can do that by [following these directions.](###Optional-Configure-AWS-Credentials-for-Amazon-Textract-usage)

1. Run the Sycamore HTTP Crawler container with an additional parameter to crawl the Sort Benchmark website:

`docker compose run sycamore_crawler_http_sort_all`

Note: If you just want to ingest one PDF instead of the whole dataset (to save time), run:

`docker compose run sycamore_crawler_http_sort_one`

Sycamore will automatically start processing the new data. The processing job is complete and the data is loaded into the index once you see log messages similar to:

`No changes at [datetime] sleeping`

3. Use the demo UI for conversational search. Using your internet browser, visit http://localhost:3000. You can interact with the demo UI while data is being added to the index, but the data won't all be available until the job is done.

* Create a new conversation. Enter the name for your conversation in the text box in the left "Conversations" panel, and hit enter or click the "add convo" icon on the right of the text box.
* Select your conversation, and then write a question into the text box in the middle panel. Hit enter.
* Ask follow up questions. You'll see the actual results from the Sycamore's hybrid search for your question in the right panel, and the conversational search in the middle panel.

Once the data has been loaded, some sample questions to ask are:

* Who are the most recent winners of the Sort Benchmark?
* What are the hardware costs for ELSAR? Return as a table.
* What is CloudSort?


## Demo: Ingest and Query Data From An Arbitrary Website

You can crawl and ingest an arbitrary website with the Sycamore default data ingestion code. However, this code is not not optimized for any given dataset, so the answer quality may vary on new websites.

Note: Processing data using the default code will send your data to OpenAI, and optionally Amazon Textract, for calls to their AI services. Consider whether this is acceptable if you are using a non-public website for testing.

1. Run the Sycamore HTTP Crawler container with an additional parameter:

```
docker compose run sycamore_crawler_http <url>
# for example
# docker compose run sycamore_crawler_http http://www.aryn.ai
```

This will crawl and download the data from the specified website. If you import aryn.ai, you can try "who are the Aryn founders?"

2. Sycamore will automatically start processing the new data into default index (which is also used in the previous example). The processing job is complete and the data is loaded into the index once you see log messages similar to:

`No changes at [datetime] sleeping`

3. Go to the demo query UI at localhost:3000. You can interact with the demo UI while data is being added to the index, but the data won't all be available until the job is done.

If you want to prepare you data with custom code, you can [use a Jupyter notebook to iterate and test it](##Use-a-Jupyter-notebook).

## Add a dataset from a S3 bucket

You can try using an arbitrary PRF and HTML data from an S3 bucket with the Sort Benchmark
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

## Use a Jupyter notebook

A [Jupyter](https://jupyter.org/) notebook is a local development environment that lets you
interact with Sycamore and experiment with different segmentations to improve the results of
processing your documents. The [local development
instructions](sycamore-local-development-example.md) will walk you through setting up Jupyter or
running it in a container, and analyzing a new set of documents.

NEED LINKS!


## Clean Up

To clean up your Sycamore resources:

1. Stop the containers

2. Run the reset script:

```
docker compose down
docker compose run reset
```

## Troubleshooting

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
