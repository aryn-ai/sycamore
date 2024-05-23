The ttviz program is a small utility for visualize the TimeTrace output for Sycamore scripts.

## Basic Usage

Compile ttviz

    make

Run a Scyamore script with TimeTrace enabled:

    TIMETRACE=/path/to/output/ poetry run python my_scamore_script.py

Run ttviz on output.

    ./ttviz /path/to/output/*

The visualization is written to `viz.png`

## Mac OS X Setup Instructions

This program depends on the `gd` utility. The easiest way to install this on the Mac is using Homebrew:

    brew install libgd

in order for the compiler to find the new library, you need to set the `CPATH` and `LIBRARY_PATH` environment variables to pick up Homebrew installed libraries. Depending on your shell and setup, the following may work

    export CPATH=$HOMEBREW_PREFIX:$CPATH
    export LIBRARY_PATH=$HOMEBREW_PREFIX:$LIBRARY_PATH

Typically `HOMEBREW_PREFIX` should be set to `/opt/homebrew`.
