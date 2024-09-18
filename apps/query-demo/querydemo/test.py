#!/usr/bin/env python

# This script is used to run the Streamlit-based Luna Query demo.
# Because of issues with Ray, we need to initialize Ray externally
# to the demo process, and then run the Streamlit demo in a subprocess.

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

ret = subprocess.run(["python", "querydemo/Query_Demo_test.py"])
print(f"Test subprocess exited {ret}", flush=True)
