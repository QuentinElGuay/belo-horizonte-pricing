import os

import boto3
import pytest
from moto import mock_aws

from mlflow import pyfunc
from src.library import serve


@pytest.fixture
def s3_boto():
    """Create an S3 boto3 client and return the client object"""

    s3 = boto3.client('s3', region_name='us-east-1')
    return s3


@mock_aws
def test_load_model_from_s3(s3_boto):
    """Test downloading a model mocking S3 with moto"""

    DEFAULT_MODEL_BUCKET = 'DEFAULT_MODEL_BUCKET'
    os.environ['DEFAULT_MODEL_BUCKET'] = DEFAULT_MODEL_BUCKET

    MODEL_NAME = 'MODEL_NAME'
    os.environ['MODEL_NAME'] = MODEL_NAME

    s3_boto.create_bucket(Bucket=DEFAULT_MODEL_BUCKET)

    with open('tests/fixtures/model/model.pkl', 'rb') as data:
        s3_boto.upload_fileobj(
            data, DEFAULT_MODEL_BUCKET, f'{MODEL_NAME}/model.pkl'
        )
    with open('tests/fixtures/model/MLmodel', 'rb') as data:
        s3_boto.upload_fileobj(
            data, DEFAULT_MODEL_BUCKET, f'{MODEL_NAME}/MLmodel'
        )

    loaded_model = serve.load_model_from_s3(MODEL_NAME)

    assert isinstance(loaded_model, pyfunc.PyFuncModel)
