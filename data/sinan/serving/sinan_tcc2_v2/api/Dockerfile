FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR /app

COPY data/sinan/serving/sinan_tcc2_v2/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app/data/sinan/serving/sinan_tcc2_v2/api", "sinan_api:app", "--host", "0.0.0.0", "--port", "8000"]
