"""WSGI entry point for PythonAnywhere (and compatible WSGI hosts).

The web app lives in ``site/`` while the scientific engine remains importable
from the project root.
"""
import sys
import os

# Ensure 'site/' is importable and project root is in path (for tumeur/)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "site"))

from app import app  # noqa: E402 — site/app.py
