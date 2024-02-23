# Repo name: arynai/sycamore-opensearch

ARG TAG=2.12.0
FROM opensearchproject/opensearch:$TAG

USER root
RUN yum install -yq openssl jq && yum clean all
USER opensearch

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
ARG DEBUG=0

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}
ENV DEBUG=${DEBUG}

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}
LABEL debug=${DEBUG}

# OpenSearch complains if these files have permissive modes.
USER root
RUN chown opensearch:opensearch \
    /usr/share/opensearch/sycamore-opensearch.sh \
    /usr/share/opensearch/config/*.yml
RUN chmod 600 /usr/share/opensearch/config/*.yml
USER opensearch

ENTRYPOINT ["/usr/share/opensearch/sycamore-opensearch.sh"]
