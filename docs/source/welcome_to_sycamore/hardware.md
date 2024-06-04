# Recommended resources to run Sycamore

## Running Locally
We recommend local deployments of Sycamore have at least 8 vCPU and 16 GB of RAM. We have tested this on a MacBook Pro with these specifications without enabling reranking, which will cause extra load on thes ystem.

Sycamore will run faster if it has more resources, primarily with a RAM/vCPU ratio of 2GB RAM/vCPU. If you see memory issues when running Sycamore, you can adjust the memory allocation further.

To make these adjustments, go to the “Settings” menu in Docker. Next, click on “Resources” and adjust the settings.

## Running on a Virtual Machine (VM)
Virtual machine deployments of Sycamore have the same vCPU and RAM resource requirements. You will need about 50GB of local disk space to install all the dependencies and containers and still have moderate free space. On AWS, we recommend:

- t3.xlarge for running Sycamore's base configuration (no reranking)
- m7.2xlarge for running Sycamore with reranking
