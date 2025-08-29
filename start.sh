#!/bin/sh

# This script runs the Streamlit dashboard.
# The Dockerfile has already set up the environment so the 'streamlit' command is available.
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
