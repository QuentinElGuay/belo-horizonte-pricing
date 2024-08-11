FROM python:3.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy local file to the working directory
COPY src/ .
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source files
COPY src/lambda.py .

CMD ["/app/lambda.lambda_handler"]
