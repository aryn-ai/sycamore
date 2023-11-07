ARG TAG=2.11.0
FROM opensearchstaging/opensearch:$TAG
USER 0
RUN yum install -y jq && yum clean all
USER 1000
COPY opensearch.yml /usr/share/opensearch/config/
COPY aryn-opensearch.sh .
ENV discovery.type=single-node
ENV DISABLE_SECURITY_PLUGIN=true
CMD ./aryn-opensearch.sh
