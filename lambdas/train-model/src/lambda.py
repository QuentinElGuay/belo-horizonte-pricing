import json
import logging
from os import getenv
import os

import awswrangler as wr
from mlflow import set_tracking_uri, set_experiment

from library.dataset import clean_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):

    RANDOM_STATE = event.get('random_state', 42)

    TRACKING_SERVER_URI = getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    set_tracking_uri(TRACKING_SERVER_URI)

    EXPERIENCE_NAME = event['experience_name']
    DATASET_PATH = event['dataset_path']
    MODEL_TYPE = event['model_type']

    match MODEL_TYPE:
        case 'simple_linear_regression':
            from library.train import \
                train_simple_linear_regression as train_model
        case 'elastic_net_regression':
            from library.train import \
                train_elastic_net_regression as train_model
        case _:
            raise ValueError(f'Invalid model type: {MODEL_TYPE}')

    # Get dataset
    df = wr.s3.read_parquet(path=DATASET_PATH)

    # Prepare data
    df = clean_dataset(df)

    variables = {
        'target': 'price',
        'categorical': ['neighborhood'],
        'numerical': ['square_foot', 'garage_places', 'rooms'],
    }

    set_experiment(EXPERIENCE_NAME)
    logger.info('Starting experiment %s', EXPERIENCE_NAME)
    
    best_run_id, best_params = train_model(df, variables, RANDOM_STATE)

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'best_run_id': best_run_id,
            'best_params': best_params

        }),
    }

    return response


if __name__ == '__main__':

    wr.config.s3_endpoint_url = 'http://localhost:9000/'
    os.environ['TRACKING_SERVER_URI'] = 'http://localhost:5000/'
    os.environ['AWS_ACCESS_KEY_ID'] = 'XeAMQQjZY2pTcXWfxh4H'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ'
    os.environ['DESTINATION_BUCKET'] = 'datasets'
    os.environ['BASE_PATH'] = 'mlops'

    event = {
        'dataset_path': 's3://datasets/mlops/belo_horizonte_price/2021-06-01/train_dataset.parquet',
        'experience_name': 'test_belo_horizonte_pricing',
        'model_type': 'elastic_net_regression',
        'split_ratio': 0.2,
    }

    lambda_handler(event, {})
