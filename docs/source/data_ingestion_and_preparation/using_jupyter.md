# Using Jupyter to create data preparation jobs

Using a [Jupyter notebook](https://jupyter.org/) makes it easy to write and iterate on custom data preparation code. By default, Sycamore includes a containerized Jupyter notebook configured with Sycamore’s data preparation libraries and other dependencies. You can easily use this to create and run scripts in a notebook or run data preparation code from the Jupyter shell. You can also [install and configure Jupyter locally](###running-jupyter-locally).

## Using the Jupyter container

### Jupyter notebook

Sycamore includes a default container with Jupyter. In order to maintain security, Jupyter uses a token to limit access to the notebook. For the first 5 minutes after launching, the Jupyter container will periodically print out instructions for connecting. If you can't find these instructions, you can get them by running:

`docker compose logs jupyter`

Then connect to the notebook using the specified URL or via the redirect file. The token is stable over restarts of the container.

The “examples” directory in Jupyter contains several example notebooks with data preparation scrips. If you write your own script, we recommend you save it in the docker_volume or bind_dir directories so it persists over container restarts. The bind_dir will enable you to save your script after container shutdown (or add files locally for the container to access), and you can find it at `../jupyter/bind_dir` on the machine running Docker.

For a tutorial on creating a data preparation script in a notebook, [click here](/tutorials/sycamore-jupyter-dev-example.md).

### Jupyter shell

If you already have custom data preparation code, the easiest way to run it is using the Juypter shell in the Jupyter container. The Jupyter container is configured with a bind direcotry, making it easy to add your script and make it accessible in the container. Once you have added your script to the bind_dir, you can [run it using the Jupyter shell](/running_a_data_preparation_job.md#using-jupyter-container).

## Running Jupyter locally

You can run Jupyter locally and load the output of your data preparation script into your Sycamore stack. The OpenSearch client configuration must match the endpoint of your Sycamore stack. For instructions on how to install and configure Jupyter locally for Sycamore, [click here](/sycamore-jupyter-dev-example.md#in-your-local-development-environment). For an example script, [click here](https://github.com/aryn-ai/sycamore/blob/main/notebooks/sycamore_local_dev_example.ipynb). 
