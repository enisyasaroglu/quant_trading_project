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
# Copy the new start-up script into the container
COPY start.sh .
# Make the new start-up script executable
RUN chmod +x ./start.sh

# Set the command to run when the container starts.
CMD ["./start.sh"]