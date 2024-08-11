import json
import logging
from typing import Any, Dict, Tuple

import awswrangler as wr
import pandas as pd

from library.dataset import split_test_dataset
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):

    DESTINATION_BUCKET = os.environ['DESTINATION_BUCKET']
    DESTINATION_FOLDER = os.environ['DESTINATION_FOLDER']

    DATASET_NAME, DATASET_DATE, DATASET_PATH = get_dataset_info(event)

    SPLIT_RATIO = event.get('split_ratio', 0.2)
    RANDOM_STATE = event.get('random_state', 42)

    if 'split_ratio' not in event:
        logger.warning('Split ratio not provided, using default value')
    logger.info('Split ratio: %s', SPLIT_RATIO)

    if DATASET_PATH.startswith('s3://'):
        # Download from S3
        df = wr.s3.read_csv(path=DATASET_PATH, sep=',')
    else:
        # Download from local file system or URI
        df = pd.read_csv(DATASET_PATH, sep=',')

    # Split dataset
    df_train, df_test = split_test_dataset(df, SPLIT_RATIO, RANDOM_STATE)

    # Upload datasets to S3
    destination_path = f's3://{DESTINATION_BUCKET}/{DESTINATION_FOLDER}'

    df_train_path = os.path.join(
        destination_path, DATASET_NAME, DATASET_DATE, 'train_dataset.parquet'
    )
    store_dataset(df_train, df_train_path)

    df_test_path = os.path.join(
        destination_path, DATASET_NAME, DATASET_DATE, 'test_dataset.parquet'
    )
    store_dataset(df_test, df_test_path)

    # Return response
    response = {
        'statusCode': 200,
        'body': json.dumps(
            {
                'training_dataset': df_train_path,
                'test_dataset': df_test_path,
                'split_ratio': SPLIT_RATIO,
            }
        ),
    }

    return response


def get_dataset_info(event: Dict[str, Any]) -> Tuple[str]:
    """Extract dataset information from Lambda event

    Args:
        event (Dict[Any]): The event dictionnary.

    Returns:
        Tuple[str]: The extracted values.
    """
    DATASET_NAME = event['dataset']['name']
    logger.info(f'Input dataset name: {DATASET_NAME}')

    DATASET_DATE = event['dataset']['date']
    logger.info(f'Input dataset date: {DATASET_DATE}')

    DATASET_PATH = event['dataset']['path']
    logger.info(f'Input dataset path: {DATASET_PATH}')

    return (DATASET_NAME, DATASET_DATE, DATASET_PATH)


def store_dataset(df: pd.DataFrame, path: str):
    """Store the dataset as parquet in S3.

    Args:
        df (pd.DataFrame): Dataset to store
        path (str): Path to store the dataset
    """
    # Upload to S3
    wr.s3.to_parquet(
        df=df,
        path=path,
    )

    logger.info('Dataset stored in %s', path)


if __name__ == '__main__':

    wr.config.s3_endpoint_url = 'http://localhost:9000/'
    os.environ['AWS_ACCESS_KEY_ID'] = 'XeAMQQjZY2pTcXWfxh4H'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ'
    os.environ['DESTINATION_BUCKET'] = 'datasets'
    os.environ['DESTINATION_FOLDER'] = 'mlops'

    event = {
        'dataset': {
            'name': 'belo_horizonte_price',
            'date': '2021-06-01',
            'path': 'data/data_kaggle_2021.csv',
        },
        'split_ratio': 0.2,
    }

    lambda_handler(event, {})
