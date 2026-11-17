FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt \
    && pip install --no-cache-dir -r requirements.txt
COPY train.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
