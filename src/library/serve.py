import logging
import os
from typing import Any, Dict

from numpy import ndarray
from pandas import DataFrame

import mlflow
from library.dataset import prepare_features

logger = logging.getLogger(__name__)


def load_model_from_tracker(model_uri: str) -> Any:
    """Load a model from the MLFlow tracker server.

    Args:
        model_uri (str): The URI of the model to load.

    Returns:
        Any: The loaded model.
    """
    logger.warning('Loading the model from the tracker server.')

    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    return model


def load_model_from_s3(model_name: str) -> Any:
    """Load a model from S3.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Any: The loaded model.
    """
    DEFAULT_MODEL_BUCKET = os.getenv('DEFAULT_MODEL_BUCKET')
    model_uri = f's3://{DEFAULT_MODEL_BUCKET}/{model_name}/model.pkl'

    logger.warning(
        'Trying to loading the default model from S3: %s.', model_uri
    )

    return mlflow.pyfunc.load_model(model_uri=model_uri)


def predict(model_name: str, X: Dict[str, Any] | DataFrame) -> ndarray:
    """Apply a model to a batch of data and return the result of the prediction.

    Args:
        model_name (str): The name of the model to apply.
        X (Dict[str, Any] | DataFrame): DataFrame or Dictionnary to apply the model to.

    Returns:
        numpy.ndarray: The predicted values.
    """
    X = prepare_features(X)

    model_uri = f'models:/{model_name}@champion'
    try:
        model = mlflow.pyfunc.load_model(model_uri)

    except ConnectionError:
        logger.warning('Unable to connect to MLFlow.')
        model = load_model_from_s3(model_name)

    return model.predict(X)
