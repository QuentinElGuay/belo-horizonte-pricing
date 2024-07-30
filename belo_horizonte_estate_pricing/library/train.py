# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from hyperopt.pyll import scope
import logging

import numpy as np
import pandas as pd
from mlflow.sklearn.utils import _get_estimator_info_tags
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

import mlflow

logger = logging.getLogger(__name__)


def tag_run(user: str = 'Quentin El Guay'):
    """Tag the run with the estimator name and class.

    Args:
        dataset: The dataset used to train the model.
        user (str): User who trained the model.
    """
    mlflow.set_tags({'user': user})
    # mlflow.log_input(dataset, context='training')


def train_simple_linear_regression(
    X: pd.DataFrame, y: pd.DataFrame
) -> Pipeline:
    """Train a  simple linear regression model on the given data.

    Args:
        X (pandas.DataFrame): Input features.
        y (pandas.Series): Target values.

    Returns:
        sklearn.pipeline.Pipeline: Trained linear regression model with DictVectorizer.
    """
    # dataset = mlflow.data.from_pandas(
    #     pd.concat([X, y]),
    #     source='https://www.kaggle.com/datasets/guilherme26/house-pricing-in-belo-horizonte',
    #     name='House Pricing in Belo Horizonte',
    #     targets='price'
    # )

    regressor = LinearRegression()

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_dict('records'), y.values, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name='Simple Linear Regression'):

        tag_run()

        pipeline = make_pipeline(
            DictVectorizer(sparse=True, dtype=int),
            TransformedTargetRegressor(
                regressor=regressor, func=np.log1p, inverse_func=np.expm1
            ),
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        metrics = {
            'rmse': root_mean_squared_error(y_val, y_pred),
            'score': pipeline.score(X_val, y_val),
        }
        mlflow.log_metrics(metrics)

        run_model = mlflow.sklearn.log_model(pipeline, 'model')

    return run_model.model_uri


# def train_regularized_regression(X:DataFrame, y:DataFrame, num_trials:int=50) -> Pipeline:
#     """Train the regularized regression models (Lasso and Ridge) on the given data.

#     Args:
#         X (pandas.DataFrame): Input features.
#         y (pandas.Series): Target values.
#         num_trials (int, optional): Number of trials for hyperparameter optimization.
#           Defaults to 50.

#     Returns:
#         Pipeline: The pipeline of the best model found during optimization.
#     """

#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     dv = DictVectorizer()
#     X_train = dv.fit_transform(X_train.to_dict('records'))
#     X_val = dv.transform(X_val.to_dict('records'))

#     def objective(params):

#         model_name, model_class = params.pop('model')
#         model = model_class(**params)

#         with mlflow.start_run(run_name=f'{model_name} Optimization', nested=True):
#             mlflow.log_params(params)

#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_val)
#             rmse_optimized = root_mean_squared_error(y_val, y_pred)
#             mlflow.log_metric('rmse', rmse_optimized)
#             # mlflow.sklearn.log_model(model, "model")

#             return {'loss': rmse_optimized, 'status': STATUS_OK}

#     search_space = {
#         'model':hp.choice('model', [('Lasso', Lasso), ('Ridge',  Ridge)]),
#         'alpha': hp.uniform('alpha', 0.001, 1.0),
#         'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 1)),
#         'random_state': 42
#     }

#     with mlflow.start_run(run_name='Regularized Linear Regression'):
#         fmin(
#             fn=objective,
#             space=search_space,
#             algo=tpe.suggest,
#             max_evals=num_trials,
#             trials=Trials(),
#             rstate=np.random.default_rng(42)
#         )

# Register the best model
# register_best_model(client, top_n)
# Logger.info("Optimization completed successfully.")
