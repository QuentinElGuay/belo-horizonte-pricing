import json
import logging
import os

import awswrangler as wr
import click
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import mlflow
from library.dataset import (
    clean_dataset,
    read_csv,
    split_test_dataset,
    store_dataset,
)


@click.group()
def cli():
    pass


@cli.command('prepare-dataset')
@click.argument('experience-name')
@click.argument('dataset-uri')
def prepare_dataset(experience_name: str, dataset_uri: str):
    """Download a dataset, split it in train and test datasets and store it in AWS S3.

    Args:
        experience_name (str): Name of the experience the dataset will be used for.
        dataset_uri (str): URI of the dataset to download.
    """
    # TODO: move before command
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL is not None:
        wr.config.s3_endpoint_url = S3_ENDPOINT_URL

    DATASET_BUCKET = os.environ['DATASET_BUCKET']
    DATASET_ROOT_FOLDER = 'datasets'
    DATASET_FOLDER_URI = f's3://{DATASET_BUCKET}/{DATASET_ROOT_FOLDER}'

    logger.info('Downloading dataset from %s', dataset_uri)
    df: pd.DataFrame = read_csv(dataset_uri)

    df_train, df_test = split_test_dataset(df, test_size=0.2, random_state=42)

    store_dataset(
        df_train, f'{DATASET_FOLDER_URI}/{experience_name}/train.parquet'
    )

    store_dataset(
        df_test, f'{DATASET_FOLDER_URI}/{experience_name}/test.parquet'
    )


@cli.command('train-model')
@click.argument('experience-name')
@click.argument('model-name')
@click.option('--dataset-uri', '-d')
def train_model(
    experience_name: str, model_name: str, dataset_uri: str = None
):
    """Train a model and store its results in MLFlow.

    Args:
        model_name (str): The name of the model to train.
        experience_name (str): Name of the experience the dataset will be used for.
        dataset_uri (str): The dataset to use to train the model.
    """
    # TODO: move before command
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL is not None:
        wr.config.s3_endpoint_url = S3_ENDPOINT_URL

    RANDOM_STATE = os.getenv('random_state', 42)

    TRACKING_SERVER_URI = os.getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    if dataset_uri:
        path = dataset_uri
    else:
        DATASET_BUCKET = os.environ['DATASET_BUCKET']
        DATASET_ROOT_FOLDER = 'datasets'
        DATASET_FOLDER_URI = f's3://{DATASET_BUCKET}/{DATASET_ROOT_FOLDER}'
        path = f'{DATASET_FOLDER_URI}/{experience_name}/train.parquet'

    match model_name:
        case 'simple_linear_regression':
            from library.train import (
                train_simple_linear_regression as train_model,
            )
        case 'elastic_net_regression':
            from library.train import (
                train_elastic_net_regression as train_model,
            )
        case _:
            raise ValueError(f'Invalid model type: {model_name}')

    # Get dataset
    df = wr.s3.read_parquet(path)

    df = clean_dataset(df)

    variables = {
        'target': 'price',
        'categorical': ['neighborhood'],
        'numerical': ['square_foot', 'garage_places', 'rooms'],
    }

    mlflow.set_experiment(experience_name)
    logger.info('Starting experiment %s', experience_name)

    best_run_id, best_params = train_model(df, variables, RANDOM_STATE)

    response = {
        'statusCode': 200,
        'body': json.dumps(
            {'best_run_id': best_run_id, 'best_params': best_params}
        ),
    }

    return response


@cli.command('evaluate-experience')
@click.argument('experience-name')
@click.option('--challengers', '-c', default=3)
@click.option(
    '--auto-promote',
    is_flag=True,
    show_default=True,
    default=False,
    help='Automatically promote the best model.',
)
@click.option('--dataset_uri', '-d')
def evaluate_experience(
    experience_name: str,
    challengers: int,
    auto_promote: bool,
    dataset_uri=None
):
    """Evaluate the runs of an experiment to select new challenger models and, optionnaly, promote a new champion.

    Args:
        experience_name (str): The name of the experiment to analyze.
        number_challengers (int): Quantity of challengers to select.
        auto_promote (bool): Whether to automatically promote the best model. Defaults to False.
        dataset_uri (str): The dataset to test the model with.
    """
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL is not None:
        wr.config.s3_endpoint_url = S3_ENDPOINT_URL

    TRACKING_SERVER_URI = os.getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    if dataset_uri:
        path = dataset_uri
    else:
        DATASET_BUCKET = os.environ['DATASET_BUCKET']
        DATASET_ROOT_FOLDER = 'datasets'
        DATASET_FOLDER_URI = f's3://{DATASET_BUCKET}/{DATASET_ROOT_FOLDER}'
        path = f'{DATASET_FOLDER_URI}/{experience_name}/test.parquet'

    # Load test_dataset
    logger.info('Loading test dataset from %s', path)
    test_df = wr.s3.read_parquet(path)
    test_df = clean_dataset(test_df)

    # Get current champion if exists
    champion_rmse = float('inf')

    try:
        champion = mlflow.pyfunc.load_model(
            f'models:/{experience_name}@champion'
        )
        pred = champion.predict(test_df)
        champion_rmse = root_mean_squared_error(test_df['price'], pred)
    except mlflow.exceptions.MlflowException as e:
        logger.info('No champion found: %s', e.message)

    # Get N best challengers
    best_rmse_runs = mlflow.search_runs(
        experiment_names=[experience_name],
        order_by=['metrics.rmse DESC'],
        max_results=challengers,
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
    if auto_promote and best_challenger_rmse < champion_rmse:
        new_champion = mlflow.register_model(
            f'runs:/{best_challenger.metadata.run_id}/model',
            experience_name,
        )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            new_champion.name, 'champion', new_champion.version
        )

    response = {
        'statusCode': 200,
        'body': json.dumps({}),
    }

    return response


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cli()
