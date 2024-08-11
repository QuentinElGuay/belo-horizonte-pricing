FROM ghcr.io/mlflow/mlflow:v2.14.3

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt && pip freeze
