"""Pure PySpark transform functions, split out of the gold-layer notebooks
so they're importable and testable without a Databricks connection.

Each module here mirrors one notebook: transforms.silver <-> 02, etc.
Notebooks import from here and stay thin -- just orchestration (reading
tables, writing tables, calling main()).
"""
