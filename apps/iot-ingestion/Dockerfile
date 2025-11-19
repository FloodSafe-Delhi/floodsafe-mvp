FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary

COPY . .

CMD ["python", "src/main.py"]
