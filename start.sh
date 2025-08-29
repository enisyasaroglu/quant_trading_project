#!/bin/sh

# This script runs the Streamlit dashboard.
# The --server.port $PORT and --server.address 0.0.0.0 flags are
# essential for the cloud platform to correctly route traffic to the app.
streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0