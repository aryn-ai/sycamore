# Repo name: arynai/sycamore-jupyter

ARG TAG=stable
FROM arynai/sycamore-importer:$TAG

# TODO: eric - remove this once we switch other containers over to non-root
# Note user/group should be 1000 to match with likely defaults on hosts.
USER root
COPY apps/jupyter/docker-root-setup.sh /root
RUN /bin/bash /root/docker-root-setup.sh
COPY apps/jupyter/sudoers /etc/sudoers
COPY apps/jupyter/fixuser.py /root/fixuser.py

# User 'app' must be id number 1000 (the default) for all to work properly.
USER app
RUN poetry run pip install notebook ipywidgets 'opensearch-py>=2.4' && rm -rf /app/.cache
RUN mkdir -p /app/work/docker_volume /app/work/bind_dir /app/work/examples
RUN touch /app/work/AAA_SEE_README_FOR_PERSISTENT_DATA_DIRECTORIES
COPY apps/jupyter/run-jupyter.sh ./
COPY apps/jupyter/README.md ./work
COPY notebooks/jupyter_dev_example.ipynb notebooks/default-prep-script.ipynb ./work/examples/
RUN perl -i -pe 's/localhost/opensearch/ if /9200/;s,tmp/sycamore/data,/app/work/docker_volume,' /app/work/examples/jupyter_dev_example.ipynb

ARG GIT_BRANCH="unknown"
ARG GIT_COMMIT="unknown"
ARG GIT_DIFF="unknown"

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}
ENV SSL=0

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}

ENTRYPOINT sudo -E /root/fixuser.py
