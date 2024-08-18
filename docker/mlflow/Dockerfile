FROM ghcr.io/mlflow/mlflow:v2.14.3

RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
