#!/bin/bash
echo "Version-Info, Sycamore Crawler HTTP Branch: ${GIT_BRANCH}"
echo "Version-Info, Sycamore Crawler HTTP Commit: ${GIT_COMMIT}"
echo "Version-Info, Sycamore Crawler HTTP Diff: ${GIT_DIFF}"

if [[ -O /app && -O /app/.scrapy ]]; then
    : # ok, proper ownership
else
    echo "ERROR: /app or /app/.scrapy has incorrect ownership"
    echo "TO FIX: docker compose run fixuser"
    exit 1
fi
case "$#-$1" in
    1-http*)
        poetry run scrapy crawl sycamore -a url="$1"
        exit 0
        ;;
    1-help|1---help)
        echo
        echo "Usage:"
        echo "# crawl starting at a url to the domain associated with that url (stripping www)"
        echo "docker compose up sycamore_crawler_http <url>"
        echo "# crawl starting at a url with a specified domain"
        echo "or docker compose up sycamore_crawler_http -a url=<url> -a domain=<domain>"
        echo "# crawl starting at a url ignoring links that don't match the prefix"
        echo "or docker compose up sycamore_crawler_http -a url=<url> -a prefix=<string>"
        exit 0
        ;;
esac

poetry run scrapy crawl sycamore "$@"
