# ---- Builder Stage ----
FROM python:3.10-slim AS builder

WORKDIR /app
RUN pip install poetry
COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.in-project true && poetry install --no-root --only main

# ---- Final Stage ----
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire project's source code
COPY . .

# --- THIS IS THE FIX ---
# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the command to run when the container starts.
# This is the most reliable way to define the start command.
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]