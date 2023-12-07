# Repo name: arynai/jupyter

ARG TAG=stable
FROM arynai/sycamore:$TAG

# TODO: eric - remove this once we switch other containers over to non-root
# Note user/group should be 1000 to match with likely defaults on hosts.
RUN groupadd --gid 1000 app
RUN useradd -d /app --uid 1000 --gid app app
RUN chown -R app:app /app

COPY jupyter/docker-apt-packages.sh ./
RUN /bin/bash docker-apt-packages.sh

COPY jupyter/sudoers /etc/sudoers
COPY jupyter/fixuser.py /fixuser.py

USER 1000
RUN poetry run pip install notebook ipywidgets 'opensearch-py>=2.4' && rm -rf /app/.cache
RUN mkdir -p /app/work/docker_volume /app/work/bind_dir /app/work/examples
RUN touch /app/work/AAA_SEE_README_FOR_PERSISTENT_DATA_DIRECTORIES
COPY jupyter/run-jupyter.sh ./
COPY jupyter/README.md ./work
COPY sycamore_local_dev_example.ipynb ./work/examples
RUN perl -i -pe 's/localhost/opensearch/ if /9200/;s,tmp/sycamore/data,/app/work/docker_volume,' /app/work/examples/sycamore_local_dev_example.ipynb

ARG GIT_BRANCH="unknown"
ARG GIT_COMMIT="unknown"
ARG GIT_DIFF="unknown"

ENV GIT_BRANCH=${GIT_BRANCH}
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_DIFF=${GIT_DIFF}

LABEL org.opencontainers.image.authors="opensource@aryn.ai"
LABEL git_branch=${GIT_BRANCH}
LABEL git_commit=${GIT_COMMIT}
LABEL git_diff=${GIT_DIFF}

ENTRYPOINT sudo -E /fixuser.py
