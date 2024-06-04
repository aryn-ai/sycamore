# Aryn Jupyter integration notes

* The directory /app/work/crawl_data contains data crawled by the sycamore crawlers

* The directory /app/work/docker_volume is a docker volume for storing Jupyter scrypts. It will
  persist if you delete the quickstart directory on your host, but is not directly accessible.

* The directory /app/work/bind_dir is a bind mount to the quickstart/jupyter/bind_dir directory on
  your host. It will be deleted if you delete the quickstart directory on your host, but is
  directly accesible on the host.

* The directory /app/work/examples contains an example from the [Aryn Quickstart local development
  guide](https://github.com/aryn-ai/quickstart/blob/main/sycamore-local-development-example.md)
