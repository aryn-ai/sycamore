#!/bin/sh
echo "Version-Info, Sycamore Crawler HTTP Branch: ${GIT_BRANCH}"
echo "Version-Info, Sycamore Crawler HTTP Commit: ${GIT_COMMIT}"
echo "Version-Info, Sycamore Crawler HTTP Diff: ${GIT_DIFF}"

poetry run scrapy crawl sycamore "$@"
