#!/usr/bin/env python

# This script is used to run the Streamlit-based Luna Query demo.
# Because of issues with Ray, we need to initialize Ray externally
# to the demo process, and then run the Streamlit demo in a subprocess.

import logging
import os
import subprocess
import time

import ray
from sycamore.executor import _ray_logging_setup

from util import ray_init

ray_init()

while not ray.is_initialized():
    print("Waiting for ray to initialize...", flush=True)
    time.sleep(1)

os.environ["EXTERNAL_RAY"] = "1"

while True:
    print("Starting streamlit process...", flush=True)
    ret = subprocess.run(["python", "-m", "streamlit", "run", "querydemo/Query_Demo.py"])
    print(f"Subprocess exited {ret}", flush=True)
