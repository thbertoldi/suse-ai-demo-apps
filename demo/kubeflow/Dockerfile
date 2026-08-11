FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY iris_lib.py otel_boot.py steps.py ./
ENTRYPOINT ["python", "-m", "steps"]
