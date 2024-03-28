# Repo name: arynai/sycamore-opensearch

FROM gradle:jdk17 AS build_ml_commons

WORKDIR /build_ml
ADD https://github.com/opensearch-project/ml-commons.git#2.12 ./ml-commons
WORKDIR /build_ml/ml-commons
RUN sed -i 's#if (os.macOsX) {#if (System.getProperty("os.arch") == "aarch64") {#' ml-algorithms/build.gradle
RUN sed -i 's#throw new OpenSearchException("GenerativeQAResponseProcessor failed in precessing response");#log.error(e); throw new OpenSearchException(e);#'\
    search-processors/src/main/java/org/opensearch/searchpipelines/questionanswering/generative/GenerativeQAResponseProcessor.java
RUN ./gradlew assemble

FROM gradle:jdk17 AS build_rps_plugin

WORKDIR /build
ADD https://github.com/aryn-ai/opensearch-remote-processor.git#2.x ./opensearch-remote-processor
WORKDIR /build/opensearch-remote-processor
RUN ./gradlew assemble

FROM opensearchproject/opensearch:2.12.0 AS install_ml_commons

USER opensearch
COPY --chown=opensearch:opensearch --from=build_ml_commons \
    /build_ml/ml-commons/plugin/build/distributions/opensearch-ml-2.12.0.0-SNAPSHOT.zip /tmp/
# opensearch-skills extends ml-commons, so we need to remove it before we can mess with ml commons.
RUN /usr/share/opensearch/bin/opensearch-plugin remove opensearch-skills
RUN /usr/share/opensearch/bin/opensearch-plugin remove opensearch-ml
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/opensearch-ml-2.12.0.0-SNAPSHOT.zip \
    https://repo1.maven.org/maven2/org/opensearch/plugin/opensearch-skills/2.12.0.0/opensearch-skills-2.12.0.0.zip

FROM install_ml_commons AS install_rps_plugin

USER opensearch
COPY --chown=opensearch:opensearch --from=build_rps_plugin \
    /build/opensearch-remote-processor/build/distributions/remote-processor-2.12.0-SNAPSHOT.zip /tmp/

RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/remote-processor-2.12.0-SNAPSHOT.zip

FROM install_rps_plugin AS opensearch

USER root
RUN yum install -yq openssl jq && yum clean all
USER opensearch

COPY --chown=opensearch:opensearch apps/opensearch/opensearch.yml /usr/share/opensearch/config/
COPY --chown=opensearch:opensearch apps/opensearch/config.yml apps/opensearch/roles_mapping.yml \
     /usr/share/opensearch/config/opensearch-security/
COPY --chown=opensearch:opensearch apps/opensearch/sycamore-opensearch.sh apps/opensearch/setup_models.py \
    /usr/share/opensearch/
COPY apps/opensearch/Dockerfile /

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

# OpenSearch complains if these files have permissive modes.
USER root
RUN chown opensearch:opensearch \
    /usr/share/opensearch/config/*.yml \
    /usr/share/opensearch/sycamore-opensearch.sh \
    /usr/share/opensearch/setup_models.py
RUN chmod 600 /usr/share/opensearch/config/*.yml
USER opensearch

ENTRYPOINT ["/usr/share/opensearch/sycamore-opensearch.sh"]
