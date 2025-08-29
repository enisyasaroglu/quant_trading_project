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

# Make the new start-up script executable
RUN chmod +x ./start.sh