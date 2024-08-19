FROM public.ecr.aws/lambda/python:3.12

RUN python -m pip install --upgrade pip

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source files
COPY library/ library/
COPY lambda.py .

CMD ["lambda.lambda_handler"]
