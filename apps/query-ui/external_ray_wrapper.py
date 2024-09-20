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
os.environ["PYTHONPATH"] = "apps/query-ui/queryui"

while True:
    print("Starting streamlit process...", flush=True)
    # Hardcode the port so you can't accidentally be using a previous run of the app
    ret = subprocess.run(
        ["python", "-m", "streamlit", "run", "apps/query-ui/queryui/Sycamore_Query.py", "--server.port", "8501"]
    )
    print(f"Subprocess exited {ret}", flush=True)
