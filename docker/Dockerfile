# Repo name: arynai/opensearch-rps
FROM gradle:jdk17 AS assemble_plugin

WORKDIR /build
COPY ./opensearch-remote-processor/ ./opensearch-remote-processor/
WORKDIR /build/opensearch-remote-processor
RUN ./gradlew assemble

FROM arynai/sycamore-opensearch:latest_rc AS opensearch
COPY --from=assemble_plugin /build/opensearch-remote-processor/build/distributions/remote-processor-2.12.0-SNAPSHOT.zip /tmp/
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/remote-processor-2.12.0-SNAPSHOT.zip
