# quickstart

You can easily get started with the Aryn Search Platform locally using Docker. If you don't have Docker already installed, visit [here](https://docs.docker.com/get-docker/). 

The quickstart will deploy the Aryn platform, consisting of four containers: Sycamore importer, Sycamore HTTP crawler, Aryn OpenSearch, and Aryn demo conversational search UI. Our images currently support XXXX hardware, and we tag each image with XXXXXX. 

The quickstart configuration of Aryn automactially runs an example workload that crawls XXXXX and loads it into the Aryn platform. This gives you a out-of-the-box example for conversational search. Please [see below](###add-your-own-data) for instructions on how to load your own data.


<GET http://sortbenchmark.org/robots.txt> (referer: None) ['cached']
arynquickstart-sycamore_crawler_http-1  | 2023-11-16 21:08:46 [scrapy.core.engine] DEBUG: Crawled (200) <GET http://sortbenchmark.org/2004_Nsort_Minutesort.pdf> (referer: None) ['cached']

The quickstart requires:

1. An OpenAI Key for LLM access. You can create an OpenAI account [here](https://platform.openai.com/signup), or if you already have one, you can retrieve your key [here](https://platform.openai.com/account/api-keys)

2. For the highest quality answers, AWS credentials for Amazon Textract and an Amazon S3 bucket for Textract input/output. The demo Sycamore processing job uses Textract for table extraction. You will accrue AWS charges for Textract usage. When setting up your environment below, you can choose to disable Textract access for simplicity, but the processing and answer quality will be lower quality.
   
    a. If you do not have an AWS account, sign up [here](https://portal.aws.amazon.com/billing/signup).  
    b. Create an Amazon S3 bucket in that account for use with Textract (e.g.  e.g. s3://username-textract-bucket). We recommend you set up bucket lifecycle rules that automatically delete files in this bucket, as the data stored here is only needed temporarily during a Sycamore data processing job.  

Now, let's get started:  

1. Download the Docker compose files for the Quickstart [here](https://github.com/aryn-ai/quickstart/tree/main/docker_compose)
UPDATE TO NEW LOCATION!  

2. Set up your Aryn Search environment:

```
export SYCAMORE_TEXTRACT_PREFIX=s3://your-bucket-name-here
export OPENAI_API_KEY=YOUR-KEY
```

a. Textract is used by default with the demo Sycamore script, and you can choose to configure it or disable it. To use it, you need to configure your AWS credentials. You can enable AWS SSO login with [these instructions](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure-profile-token.html#sso-configure-profile-token-auto-sso), or you can use other methods to set up AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and if needed AWS_SESSION_TOKEN. 

```
aws sso login --profile YOUR-PROFILE-NAME
eval "$(aws configure export-credentials --format env --profile YOUR-PROFILE-NAME)"
```

You can verify it is working by running:

```
aws s3 ls --profile YOUR-PROFILE-NAME
```
You should see the bucket you created for $SYCAMORE_TEXTRACT_PREFIX.  

If you do not want to use Textract in your Sycamore job, then you can disable it by:

```
export ENABLE_TEXTRACT=false
```

3. Start the Docker service
   a. On MacOS or Windows, start Docker desktop
   b. On Linux, if you used your local package manager, it should be started

4. Start Aryn Search
In the directory where you downloaded the Docker compose files, run:

```
docker compose up 
```


To run the quickstart:

```
cd path/to/quickstart/docker_compose
Set OPENAI_API_KEY, e.g. export OPENAI_API_KEY=sk-...XXXXX
NEED TO SETUP AWS CREDENTIALS
docker compose up
```

You can then visit http://localhost:3000/ for conversational search on the sample dataset.

### Add your own data

You can create additional indexes in the quickstart with your own data and have conversational search on it. The quickstart includes a sample Sycamore script that loads and prepares an example dataset from https://sortbenchmark.org. You can also have Aryn use this script to process your own data by configuring the quickstart to ingest it:

STEPS TO DO THIS

Please note that the sample Sycamore script was created to process the data found in the Sort Benchmark dataset and not optimized for preparing your private data for search. We recommend iterating on the Sycamore script to find the best way to prepare and enrich your data for the best quality results.

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





