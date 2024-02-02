# Loading data into Sycamore

You can load data into Sycamore using crawlers, copying a local file, or using the Sycamore data preparation libraries directly.


## Using a crawler
You can use a crawler to automatically ingest data from a data source into Sycamore. If invoked to load data again from the same location, it will only load new or changed data.

Once data is crawled, Sycamore will use the default data preparation code to process the data, create vector embeddings, and load the data in Sycamore’s indexes.

### Load data from Amazon S3

To use the S3 crawler, run this command and specify your S3 bucket (and optional folder):

```docker run -v crawl_data:/app/.data/.s3 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN crawler_s3 [your-bucket-name]/[your-folder-name]```

You can provide your AWS keys as arguments in this command, SSO, or the other ways the CLI resolves AWS credentials.

### Load data from websites

The Sycamore HTTP crawler is based on scrapy, a framework for writing web crawlers. The crawler uses scrapy's RFC2616 caching in order to reduce the load from crawling.

To use the HTTP crawler, run:

`docker run -v crawl_data:/app/.data/.scrapy crawler_http [URL]`

## Load PDFs from local machine

If you have local PDF or HTML files to load into Sycamore, you can copy them to the directory where the HTTP crawler saves the PDF or HTML files, respectively. Adding files to this directory will trigger the Sycamore-Importer to process them.

To copy a local PDF file to this directory, run:

`docker cp . [name-of-your-Sycamore-Importer-Container]:/app/.scrapy/downloads/pdf`

To copy a local HTML file to this directory, run:

`docker cp . [name-of-your-Sycamore-Importer-Container]:/app/.scrapy/downloads/html`


## Use data preparation libraries to load data

You can write data preparation jobs using the Sycamore libraries direclty and load this data into your Sycamore stack. For more information, visit LINK HERE.
