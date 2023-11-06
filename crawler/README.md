# crawler

Crawler component

## Overview

### HTTP crawler

The Sycamore HTTP crawler is based on [scrapy](http://scrapy.org), a framework for writing web
crawlers.  We override the crawl class to follow links within a single domain starting from a
single root and store those files locally to match with how the Sycamore importer wants them.  The
crawler uses scrapy's RFC2616 caching in order to reduce the load from crawling.

## Branches

* main: releaseable quality work
* dev-_username_*: checkpoints for sharing interim work

## Build

### To create the volume
docker volume create crawl_data

### To update the http crawler
docker build -t crawler_http Dockerfile.http

### To update the S3 crawler, from root directory run
docker build -t crawler_s3 -f s3/Dockerfile .

### To use the http crawler
docker run -v crawl_data:/app/.data/.scrapy crawler_http

### To use s3 crawler
docker run -v crawl_data:/app/.data/.s3 -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY crawler_s3

### To look at the data
docker run -it -v crawl_data:/app/.data ubuntu bash
