#!/bin/bash
set -eu
set -x

export DEBIAN_FRONTEND=noninteractive

apt update
apt -y install --no-install-recommends poppler-utils tesseract-ocr python3-poetry gcc python3-dev
