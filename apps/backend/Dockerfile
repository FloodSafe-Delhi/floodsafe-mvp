FROM python:3.11-slim

WORKDIR /app

# system deps for psycopg2/postgis and geospatial libs
RUN apt-get update && \
    apt-get install -y build-essential gcc libpq-dev libgeos-dev && \
    rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy app code
COPY . /app

# Ensure Python sees /app (so 'import src' works)
ENV PYTHONPATH=/app

# runtime uploads
RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
