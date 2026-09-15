"""Shared pytest fixtures for testing transforms/ locally -- no Databricks
connection needed.

Makes `transforms` importable when pytest is run from the repo root (it
usually already is, but this keeps things working regardless of pytest's
import mode / rootdir detection).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.getOrCreate()
