# crawler

Crawler component

## Branches

* main: releaseable quality work
* dev-_username_*: checkpoints for sharing interim work

## Build

```
# once
docker volume create crawl_data
# when you update the source code
docker build crawler .
# when you want it to crawl
docker run -v crawl_data:/app/.scrapy crawler
# when you want to look at the data
docker run -it -v crawl_data:/app/.scrapy crawler bash
```
