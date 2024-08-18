FROM public.ecr.aws/lambda/python:3.12

# Install requirements
COPY lambdas/serve-model/requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source files
COPY library/library library/
COPY lambdas/serve-model/src/lambda.py .

CMD ["lambda.lambda_handler"]
