#!/bin/bash
# Fallback: set env var to fix protobuf/Streamlit compatibility
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
exec streamlit run app.py "$@"
