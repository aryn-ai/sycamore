FROM opensearchstaging/opensearch:2.12.0

COPY remote-processor-2.12.0-SNAPSHOT.zip /tmp/
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/remote-processor-2.12.0-SNAPSHOT.zip

COPY opensearch.yml /usr/shars/opensearch/config/

ENV DISABLE_SECURITY_PLUGIN=true
ENV DISABLE_INSTALL_DEMO_CONFIG=true