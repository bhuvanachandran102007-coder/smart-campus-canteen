
FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements-api.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]