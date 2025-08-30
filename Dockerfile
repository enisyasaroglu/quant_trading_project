# ---- Builder Stage ----
# Use a full Python image to build the virtual environment
FROM python:3.10-slim AS builder

# Set the working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the files needed to install dependencies
COPY poetry.lock pyproject.toml ./

# Install dependencies into a virtual environment using the correct flag
RUN poetry config virtualenvs.in-project true && poetry install --no-root --only main


# ---- Final Stage ----
# Use a slim Python image for the final, smaller application image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Set the PATH to include the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire project's source code
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the command to run when the container starts.
# This is the most reliable way to define the start command and avoids parsing issues.
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
