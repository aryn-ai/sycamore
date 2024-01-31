# Recommended resources to run Sycamore 

## Running Locally 
Local deployments of Sycamore need at least 2 vCPU, 4 GB of RAM, and 4GB of Swap. Sycamore will run faster if it has more resources, primarily with a RAM/vCPU ratio of 2GB RAM/vCPU. If you see memory issues when running Sycamore, you can adjust the memory allocation further. 

To make these adjustments, go to the “Settings” menu in Docker. Next, click on “Resources” and adjust the settings. 

## Running on a Virtual Machine (VM) 
Virtual machine deployments of Sycamore have the same vCPU, RAM, Swap resource requirements, you will need about 50GB of local disk space to install all the dependencies and containers and still have moderate free space. On AWS those minimal requirements map to a t2.medium, but we recommend a t2.large or t2.xlarge to provide better performance. 
