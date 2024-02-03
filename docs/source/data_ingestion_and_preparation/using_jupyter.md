# Using Jupyter to create data preparation jobs

It can be useful to use a [Jupyter notebook](https://jupyter.org/) to write and iterate on custom data preparation code. By default, Sycamore includes a Jupyter notebook configured with Sycamore’s data preparation libraries and other dependencies, and you can easily use this to create scripts in a notebook or run data preparation code from the Jupyter shell. You can also install and configure Jupyter with the Sycamore libraries locally.

## Using the Jupyter container

### Jupyter notebook

Sycamore launches a default container with Jupyter. In order to maintain security, Jupyter uses a token to limit access to the notebook. For the first 5 minutes the Jupyter container will periodically print out instructions for connecting. If you can't see them, you can get them by running:

`docker compose logs jupyter`

Then connect to the notebook using the specified URL or via the redirect file. The token is stable over restarts of the container.

The “examples” directory contains several example notebooks with data preparation scrips. If you write your own script, we recommend you save it in the docker_volume or bind_dir directories so it persists over container restarts. The bind_dir will enable you to save your script after container shutdown (or add files locally for the container to access), and you can find it on  XXXX NEED LOCATION on the machine running Docker.

When you run a notebook cell that loads OpenSearch, it will load the output of your script into the vector and keyword indexes in your Syamore stack.

For a tutorial on creating a data preparation script, click here.

### Jupyter shell

If you already have custom data preparation code, the easiest way to run it is using the Juypter shell in the Jupyter container. The Jupyter container is configured with a bind_dir, making it easy to add your script and make it accessible in the container. Once you have added your script to the bind_dir, you can run it using the Jupyter shell:

NEED INSTRUCTIONS ON HOW TO DO THIS.

## Running Jupyter locally

You can run Jupyter locally and load the output of your data preparation script into your Sycamore stack. The OpenSearch client configuration must match the endpoint of your Sycamore stack. For instructions on how to install and configure Jupyter locally for Sycamore, click here. For an example script, click here. NEED LINK
