FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway uses PORT env var; fallback to 8000 for local dev
ENV PORT=8000

# Use shell form to expand $PORT variable
CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT
