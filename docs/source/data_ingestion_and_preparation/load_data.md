# Loading data into Sycamore

You can load data into Sycamore using crawlers, copying a local file, or using the Sycamore data preparation libraries directly.


## Using a crawler
You can use a crawler to automatically ingest data from a data source into Sycamore. If invoked to load data again from the same location, it will only load new or changed data.

Once data is crawled, Sycamore will use the default data preparation code to process the data, create vector embeddings, and load the data in Sycamore’s indexes.

### Load data from Amazon S3

To use the S3 crawler, run this command and specify your S3 bucket (and optional folder):

```
docker compose run sycamore_crawler_s3 [your-bucket-name] [optional-folder-name]

#example
docker compose run sycamore_crawler_s3 aryn-public sort-benchmark/pdf
```

If AWS credentials are required to access the S3 bucket, make sure they are properly configured before running the crawler.

### Load data from websites

The Sycamore HTTP crawler is based on scrapy, a framework for writing web crawlers. The crawler uses scrapy's RFC2616 caching in order to reduce the load from crawling.

To use the HTTP crawler, run:

```
docker compose run sycamore_crawler_http [URL]

# Download all files from aryn.ai reachable from www.aryn.ai (www is auto-removed)
docker compose run sycamore_crawler_http http://www.aryn.ai

# Download files starting from url=https://bair.berkeley.edu/blog that are under https://bair.berkeley.edu/blog/2023
# Note: if there is a 301 redirect from a URL starting with prefix to a file outside, the file will be downloaded.
docker compose run sycamore_crawler_http -a url=https://bair.berkeley.edu/blog -a prefix=https://bair.berkeley.edu/blog/2023
```

## Load PDFs from local machine

If you have local PDF or HTML files to load into Sycamore, you can copy them to the directory where the HTTP crawler saves the PDF or HTML files, respectively. Adding files to this directory will trigger the Sycamore-Importer to process them.

To copy a local PDF file, run:

`docker compose cp your-file.pdf importer:/app/.scrapy/downloads/pdf`

To copy a local HTML file, run:

`docker compose cp your-file.html importer:/app/.scrapy/downloads/html`


## Use data preparation libraries to load data

You can write data preparation jobs [using the Sycamore libraries](./installing_sycamore_libraries_locally.md) directly or [using Jupyter](./using_jupyter.md) and [load this data into your Sycamore stack](./running_a_data_preparation_job.md).
