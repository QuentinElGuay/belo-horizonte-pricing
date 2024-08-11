import json
import logging
from os import getenv
import os

import awswrangler as wr
import mlflow
from sklearn.metrics import root_mean_squared_error

from library.dataset import clean_dataset


def lambda_handler(event, context):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    TRACKING_SERVER_URI = getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    # Get experience name
    EXPERIMENT_NAME = 'regression_belo_horizonte_2021'
    MODEL_NAME = 'belo_horizonte_price_regression'
    NUMBER_CHALLENGERS = 3
    TEST_DATASET_PATH = 's3://datasets/mlops/belo_horizonte_price/2021-06-01/test_dataset.parquet'

    # Load test_dataset
    test_df = wr.s3.read_parquet(path=TEST_DATASET_PATH)
    test_df = clean_dataset(test_df)

    # Get current champion if exists
    champion_rmse = float('inf')

    try:
        champion = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}@champion')
        pred = champion.predict(test_df)
        champion_rmse = root_mean_squared_error(test_df['price'], pred)
    except mlflow.exceptions.MlflowException as e:
        logger.info('No champion found: %s', e.message)

    # Get N best challengers
    best_rmse_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=['metrics.rmse DESC'],
        max_results=NUMBER_CHALLENGERS,
    )

    best_challenger = None
    best_challenger_rmse = float('inf')
    best_run_ids = best_rmse_runs['run_id']
    for run_id in best_run_ids:
        challenger = mlflow.pyfunc.load_model(f'runs:/{run_id}/model')
        pred = challenger.predict(test_df)
        challenger_rmse = root_mean_squared_error(test_df['price'], pred)
        if challenger_rmse < best_challenger_rmse:
            best_challenger_rmse = challenger_rmse
            best_challenger = challenger

    # Promote best model
    if best_challenger_rmse < champion_rmse:
        new_champion = mlflow.register_model(
            f'runs:/{best_challenger.metadata.run_id}/model',
            MODEL_NAME,
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            new_champion.name, 'champion', new_champion.version)

    response = {
        'statusCode': 200,
        'body': json.dumps({}),
    }

    return response

if __name__ == '__main__':

    wr.config.s3_endpoint_url = 'http://localhost:9000/'
    os.environ['TRACKING_SERVER_URI'] = 'http://localhost:5000/'
    os.environ['AWS_ACCESS_KEY_ID'] = 'XeAMQQjZY2pTcXWfxh4H'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ'

    EXPERIMENT_NAME = 'regression_belo_horizonte_2021'
    MODEL_NAME = 'belo_horizonte_price_regression'
    NUMBER_CHALLENGERS = 3
    TEST_DATASET_PATH = 's3://datasets/mlops/belo_horizonte_price/2021-06-01/test_dataset.parquet'

    event = {
        'dataset_path': 's3://datasets/mlops/belo_horizonte_price/2021-06-01/train_dataset.parquet',
        'experience_name': 'test_belo_horizonte_pricing',
        'model_type': 'elastic_net_regression',
        'split_ratio': 0.2,
    }

    lambda_handler(event, {})
