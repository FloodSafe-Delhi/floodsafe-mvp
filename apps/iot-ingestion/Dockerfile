FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn

COPY . .

CMD ["python", "src/main.py"]
