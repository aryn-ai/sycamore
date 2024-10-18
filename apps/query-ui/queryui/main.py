#!/usr/bin/env python

# This script is used to run the Streamlit-based Luna Query UI.
# Because of issues with Ray, we need to initialize Ray externally
# to the demo process, and then run the Streamlit UI in a subprocess.

import argparse
import logging
import os
import subprocess
import time

import ray

from sycamore.executor import _ray_logging_setup


def ray_init(**ray_args):
    """Used to initialize Ray before running the Streamlit app."""
    if ray.is_initialized():
        return

    if "logging_level" not in ray_args:
        ray_args.update({"logging_level": logging.INFO})
    if "runtime_env" not in ray_args:
        ray_args["runtime_env"] = {}
    if "worker_process_setup_hook" not in ray_args["runtime_env"]:
        ray_args["runtime_env"]["worker_process_setup_hook"] = _ray_logging_setup
    ray.init(**ray_args)

    while not ray.is_initialized():
        print("Waiting for ray to initialize...", flush=True)
        time.sleep(1)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--exec-mode", type=str, choices=["ray", "local"], default="ray", help="Configure Sycamore execution mode."
    )
    argparser.add_argument("--chat", action="store_true", help="Only show the chat demo pane.")
    argparser.add_argument(
        "--index", help="OpenSearch index name to use. If specified, only this index will be queried."
    )
    argparser.add_argument("--cache-dir", type=str, help="Query execution cache dir.")
    argparser.add_argument(
        "--llm-cache-dir", type=str, default="llm_cache", help="LLM query cache dir. Defaults to ./llm_cache."
    )
    argparser.add_argument(
        "--trace-dir", type=str, default="traces", help="Directory to store query traces. Defaults to ./traces."
    )
    args = argparser.parse_args()

    if args.chat:
        cmdline = ["python", "-m", "streamlit", "run", "queryui/pages/Chat.py"]
    else:
        cmdline = ["python", "-m", "streamlit", "run", "queryui/Sycamore_Query.py"]

    cmdline_args = []

    if args.index:
        cmdline_args.extend(["--index", args.index])

    if args.cache_dir:
        if not args.cache_dir.startswith("s3://"):
            cache_dir = os.path.abspath(args.cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = args.cache_dir
        cmdline_args.extend(["--cache-dir", cache_dir])

    if args.llm_cache_dir:
        if not args.llm_cache_dir.startswith("s3://"):
            llm_cache_dir = os.path.abspath(args.llm_cache_dir)
            os.makedirs(llm_cache_dir, exist_ok=True)
        else:
            llm_cache_dir = args.llm_cache_dir
        cmdline_args.extend(["--llm-cache-dir", llm_cache_dir])

    if args.trace_dir:
        if not args.llm_cache_dir.startswith("s3://"):
            trace_dir = os.path.abspath(args.trace_dir)
        else:
            trace_dir = args.trace_dir
        cmdline_args.extend(["--trace-dir", trace_dir])

    if args.exec_mode == "ray":
        ray_init()
    elif args.exec_mode == "local":
        cmdline_args.extend(["--local-mode"])
    while True:
        print("Starting streamlit process...", flush=True)
        # Streamlit requires the -- separator to separate streamlit arguments from script arguments.
        ret = subprocess.run(cmdline + ["--"] + cmdline_args, check=True)
        print(f"Subprocess exited {ret}", flush=True)


if __name__ == "__main__":
    main()
