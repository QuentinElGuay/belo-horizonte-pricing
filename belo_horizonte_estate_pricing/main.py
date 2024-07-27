from os import getenv
import logging
from typing import Dict
from matplotlib import pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from library.dataset import get_dataset
from library.train import (
    train_simple_linear_regression,
)  ##, train_regularized_regression

logger = logging.getLogger(__name__)

TRACKING_SERVER_URI = getenv('TRACKING_SERVER_URI', 'http://localhost:5000')
RANDOM_STATE = 42


def main(experiment_name: str, dataset_path: str):

    logging.basicConfig(level=logging.INFO)
    np.random.seed(RANDOM_STATE)

    # Get dataset
    df, variables = get_dataset(dataset_path)

    # Split the original dataset in training and test datasets for future test.
    X_train, X_test, y_train, y_test = train_test_split(
        df[variables['numerical'] + variables['categorical']],
        df[variables['target']],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(experiment_name)
    logger.info('Starting experiment %s', experiment_name)

    # Train model(s)
    model_uri = train_simple_linear_regression(X_train, y_train)

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

    # Load model
    model_uri = f'models:/{MODEL_NAME}/{registered_model.version}'
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Predict
    inputs = {
        'neighborhood': 'Buritis',
        'square-foot': 102.0,
        'rooms': 3.0,
        'garage-places': 2.0,
    }

    def prepare_features(inputs: Dict[str, str | int | float]):
        features = {}
        features['neighborhood'] = str(inputs['neighborhood'])
        features['square_foot'] = inputs['square-foot']
        features['rooms'] = inputs['rooms']
        features['garage_places'] = inputs['garage-places']
        return features

    preds = model.predict(prepare_features(inputs))

    # y_pred = model.predict(X_test.to_dict('records'))
    # logger.info('RMSE on the test dataset: %s', root_mean_squared_error(y_test, y_pred))


if __name__ == '__main__':

    main('belo-horizonte-test-kaggle_2021', 'data/data_kaggle_2021.csv')
