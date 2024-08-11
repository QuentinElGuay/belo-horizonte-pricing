FROM python:3.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install any necessary dependencies
RUN apt-get update && \
    apt-get install -y zip && \
    rm -rf /var/lib/apt/lists/*

# Copy local file to the working directory
COPY library/ .
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt -t /opt/python/

# Set the CMD to zip the installed packages into a layer
# change the `requests-layer` to the LAYER_NAME variable as per create_layer.sh file

CMD cd /opt && zip -r9 /app/requests-layer.zip .
