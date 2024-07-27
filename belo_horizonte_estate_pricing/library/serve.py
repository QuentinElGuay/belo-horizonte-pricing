import mlflow


model = mlflow.pyfunc.load_model('models:/trip_duration/staging')

from typing import Any, Dict
from numpy import expm1
from pandas import DataFrame
from sklearn.pipeline import Pipeline


def predict_batch(X: DataFrame, pipeline: Pipeline) -> DataFrame:
    """Predict the target variable for a batch of data.

    Args:
        X (DataFrame): Bath of data to apply the model to.
        model (str): Path to the pipeline.

    Returns:
        numpy.ndarray: The predicted values.
    """

    pred = pipeline.predict(X.to_dict('records'))
    return expm1(pred)


def predict_single(features: Dict[str, Any], pipeline: Pipeline) -> float:
    """Predict the target variable for a single data point.

    Args:
        X (DataFrame): Data point to apply the model to.
        model (str): Path to the pipeline.

    Returns:
        float: The predicted value.
    """

    return expm1(pipeline.predict(features)[0])
