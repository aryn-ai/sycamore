# Repo name: arynai/sycamore-opensearch

ARG TAG=2.11.0
FROM opensearchproject/opensearch:$TAG
COPY opensearch/opensearch.yml /usr/share/opensearch/config/
COPY opensearch/config.yml opensearch/roles_mapping.yml \
     /usr/share/opensearch/config/opensearch-security/
COPY opensearch/sycamore-opensearch.sh /usr/share/opensearch/
COPY opensearch/Dockerfile /

ENV discovery.type=single-node
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

USER root
RUN yum install -yq openssl jq && yum clean all
RUN chown 1000:1000 \
    /usr/share/opensearch/sycamore-opensearch.sh \
    /usr/share/opensearch/config/*.yml
RUN chmod 600 /usr/share/opensearch/config/*.yml
USER opensearch

ENTRYPOINT ["/usr/share/opensearch/sycamore-opensearch.sh"]
