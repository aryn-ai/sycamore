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

You can provide your AWS keys as arguments in this command, SSO, or the other ways the CLI resolves AWS credentials. For instance, you can manually add these variables:

```-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN```

The `-v` specifies the volume being mounted to store crawled documents.

### Load data from websites

The Sycamore HTTP crawler is based on scrapy, a framework for writing web crawlers. The crawler uses scrapy's RFC2616 caching in order to reduce the load from crawling.

To use the HTTP crawler, run:

```
docker compose run sycamore_crawler_http [URL]

#example
docker compose run sycamore_crawler_http http://www.aryn.ai
```


## Load PDFs from local machine

If you have local PDF or HTML files to load into Sycamore, you can copy them to the directory where the HTTP crawler saves the PDF or HTML files, respectively. Adding files to this directory will trigger the Sycamore-Importer to process them.

To copy a local PDF file, run:

`docker cp . [name-of-your-Sycamore-Importer-container]:/app/.scrapy/downloads/pdf`

To copy a local HTML file, run:

`docker cp . [name-of-your-Sycamore-Importer-container]:/app/.scrapy/downloads/html`


## Use data preparation libraries to load data

You can write data preparation jobs [using the Sycamore libraries](/installing_sycamore_libraries_locally.md) direclty or [Jupyter](/using_jupyter.md) and [load this data into your Sycamore stack](/running_a_data_preparation_job.md). 
