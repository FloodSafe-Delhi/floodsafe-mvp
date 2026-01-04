FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg2 and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy ML models and data (required for embedded ML)
COPY models/ ./models/
COPY data/ ./data/

# Copy other necessary files
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Railway uses PORT env var; fallback to 8000 for local dev
ENV PORT=8000

# Enable ML by default (can be overridden via env)
ENV ML_ENABLED=true

# Use shell form to expand $PORT variable
CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT
