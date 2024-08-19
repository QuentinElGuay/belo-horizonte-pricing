FROM public.ecr.aws/lambda/python:3.12

# Set the working directory in the container
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source files
COPY library/ library/
COPY lambda_handler.py .

CMD ["lambda_handler.lambda_handler"]
