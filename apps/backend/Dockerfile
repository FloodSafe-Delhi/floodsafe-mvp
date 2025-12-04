FROM python:3.11-slim

WORKDIR /app

# Install OS dependencies required for psycopg2 & Geo libraries
RUN apt-get update && \
    apt-get install -y build-essential gcc libpq-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code (build context is apps/backend)
COPY . /app

# Ensure Python can import package `src` (parent /app must be on PYTHONPATH)
ENV PYTHONPATH=/app

# Create uploads folder
RUN mkdir -p /app/uploads

EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
