#!/usr/bin/env python3
#
# Credential helper to enable people to store fine grained access tokens in ssh environment
# variables for use on a shared instance.
# git config --global credential.helper ..../git-credentials-from-env.py
# git config --global credential.useHttpPath true

import sys
import logging
import os

if len(sys.argv) != 2 or sys.argv[1] != "get":
    exit(0)

d = {}

for line in sys.stdin:
    if line == "\n":
        break
    p = line.rstrip().split("=")  # Fix the missing split
    if len(p) != 2:
        logging.error(f"{__file__}: unable to parse {line}")
        continue
    d[p[0]] = p[1]

if d.get("host", "") == "github.com" and "aryn-ai" in d.get("path", ""):
    assert "ARYN_GITHUB_USER" in os.environ
    assert "ARYN_GITHUB_KEY" in os.environ
    print("protocol=https")
    print("host=github.com")
    print(f"username={os.environ['ARYN_GITHUB_USER']}")
    print(f"password={os.environ['ARYN_GITHUB_KEY']}")
    logging.error(f"git-credentials-from-env helper: Aryn github.com was used for {d['path']}")
    exit(0)

if "CUSTOMER_USER" in os.environ and "CUSTOMER_KEY" in os.environ:
    print("protocol=https")
    print("host=github.com")
    print(f"username={os.environ['CUSTOMER_USER']}")
    print(f"password={os.environ['CUSTOMER_KEY']}")
    logging.error(f"git-credentials-from-env helper: Customer user was used for {d['path']}")
    exit(0)

logging.error(f"WARNING from {__file__}: Unable to find CUSTOMER_USER and CUSTOMER_KEY in environ.")
logging.error(f"WARNING since the helper was enabled, this is probably an error.")
exit(0)
