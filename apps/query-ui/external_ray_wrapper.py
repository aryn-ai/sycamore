import os
import subprocess
import time

import ray

from sycamore.executor import sycamore_ray_init

sycamore_ray_init()

while not ray.is_initialized():
    print("Waiting for ray to initialize...", flush=True)
    time.sleep(1)

os.environ["EXTERNAL_RAY"] = "1"

while True:
    print("Starting streamlit process...", flush=True)
    ret = subprocess.run(["python", "-m", "streamlit", "run", "apps/query-ui/queryui/Sycamore_Query.py"])
    print(f"Subprocess exited {ret}", flush=True)
