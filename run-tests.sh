#!/usr/bin/env bash
# Run the test suite locally, inside WSL (Ubuntu). See README for setup.
#
# Windows can't host this: PySpark's JVM can't open a loopback pipe on this
# machine (sun.nio.ch.UnixDomainSockets -> Invalid argument), so the local
# environment lives in WSL instead.
#
#   ./run-tests.sh              # whole suite
#   ./run-tests.sh tests/test_fact_match.py
set -e

export JAVA_HOME="$HOME/jdk17"
export PATH="$JAVA_HOME/bin:$PATH"

# driver and workers must be the SAME python, or Spark aborts with a version
# mismatch -- Ubuntu's system python3 is not the venv's
export PYSPARK_PYTHON="$HOME/venvs/stryktipset/bin/python"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"

export PYTHONDONTWRITEBYTECODE=1

exec "$PYSPARK_PYTHON" -m pytest "${@:-tests}" -p no:cacheprovider
