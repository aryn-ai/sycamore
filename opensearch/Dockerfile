# Repo name: arynai/sycamore-opensearch

ARG TAG=2.11.0
FROM opensearchproject/opensearch:$TAG
USER 0
RUN yum install -y jq && yum clean all
USER 1000
COPY opensearch/opensearch.yml /usr/share/opensearch/config/
COPY opensearch/sycamore-opensearch.sh .
ENV discovery.type=single-node
ENV DISABLE_SECURITY_PLUGIN=true
ENV DISABLE_INSTALL_DEMO_CONFIG=true

ARG GIT_BRANCH="main"
ARG GIT_COMMIT="unknown"
ARG GIT_DIFF="unknown"

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}

CMD ./sycamore-opensearch.sh
