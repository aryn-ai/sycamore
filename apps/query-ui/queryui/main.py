#!/usr/bin/env python

# This script is used to run the Streamlit-based Luna Query UI.
# Because of issues with Ray, we need to initialize Ray externally
# to the UI process, and then run the Streamlit UI in a subprocess.

import os
import subprocess
import time

import ray

from util import ray_init

ray_init()

while not ray.is_initialized():
    print("Waiting for ray to initialize...", flush=True)
    time.sleep(1)

os.environ["EXTERNAL_RAY"] = "1"

while True:
    print("Starting streamlit process...", flush=True)
    ret = subprocess.run(["python", "-m", "streamlit", "run", "queryui/Sycamore_Query.py"])
    print(f"Subprocess exited {ret}", flush=True)
