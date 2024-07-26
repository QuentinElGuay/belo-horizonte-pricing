FROM ghcr.io/mlflow/mlflow:v2.14.3

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip freeze
