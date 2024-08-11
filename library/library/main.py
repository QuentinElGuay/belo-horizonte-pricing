import logging
from os import getenv
from typing import Dict

import numpy as np

import mlflow
from library.dataset import get_dataset, split_test_dataset
from library.serve import predict
from library.train import train_simple_linear_regression

logger = logging.getLogger(__name__)

TRACKING_SERVER_URI = getenv('TRACKING_SERVER_URI', 'http://localhost:5000')
RANDOM_STATE = 42


def main(experiment_name: str, dataset_path: str):

    logging.basicConfig(level=logging.INFO)
    np.random.seed(RANDOM_STATE)

    # Get dataset
    df, variables = get_dataset(dataset_path)

    # Split the original dataset in training and test datasets for future test.
    df_train, df_test = split_test_dataset(df, 0.2, RANDOM_STATE)

    # Start experience
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(experiment_name)
    logger.info('Starting experiment %s', experiment_name)

    # Train model(s)
    model_uri = train_simple_linear_regression(df_train, variables)

    # Register best run model
    best_rmse_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=['metrics.rmse DESC'],
        max_results=10,
    )

    best_rmse_run_id = best_rmse_runs.loc[0, 'run_id']
    model_uri = f'runs:/{best_rmse_run_id}/model'

    MODEL_NAME = 'belo_horizonte_estate_pricing'
    registered_model = mlflow.register_model(
        model_uri=model_uri, name=MODEL_NAME
    )

    registered_model = mlflow.register_model(
        model_uri=model_uri, name=MODEL_NAME, tags=[]
    )

    # Predict
    inputs = {
        'admin-fee': 0.0,
        'neighborhood': 'Buritis',
        'square-foot': 102.0,
        'rooms': 3.0,
        'garage-places': 2.0,
    }

    model_uri = f'models:/{MODEL_NAME}/{registered_model.version}'
    predict(registered_model.name, inputs)


if __name__ == '__main__':

    main('belo-horizonte-test-kaggle_2021', 'data/data_kaggle_2021.csv')
