"""Pytest config: make `src` importable when running tests from project root."""
import os
import sys

# Add the project root (parent of tests/) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
