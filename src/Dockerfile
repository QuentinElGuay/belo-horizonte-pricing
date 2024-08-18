FROM python:3.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app

RUN python -m pip install --upgrade pip

# Copy local file to the working directory
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY library/ library/
COPY main.py .

ENTRYPOINT [ "python", "main.py" ]
CMD ["--help"]
