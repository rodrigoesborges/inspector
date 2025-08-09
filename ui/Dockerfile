# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS deps
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
# Use BuildKit cache for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8998

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8998", "--server.address='0.0.0.0'","server.enableCORS=false","server.enableXsrfProtection=false","--server.headless", "true"]
