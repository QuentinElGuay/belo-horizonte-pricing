import logging
from typing import Dict

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline

import mlflow

logger = logging.getLogger(__name__)


def tag_run(user: str = 'Quentin El Guay'):
    """Tag the run with the estimator name and class.

    Args:
        dataset: The dataset used to train the model.
        user (str): User who trained the model.
    """
    mlflow.set_tags({'mlflow.user': user})
    # mlflow.log_input(dataset, context='training')


def train_simple_linear_regression(
    df: pd.DataFrame,
    variable_descriptions: Dict[str, list[str]],
    random_state: int = 42,
) -> Pipeline:
    """Train a simple linear regression model on the given data.

    Args:
        df (pandas.DataFrame): Input data.
        variable_descriptions (Dict[str, list[str]]): The description of the variables.
            It should contain the following keys: 'categorical', 'numerical' and 'target'.
        random_state: random state to use for reproducibility.

    # Returns:
    #     sklearn.pipeline.Pipeline: Trained linear regression model with DictVectorizer.
    """
    from sklearn.linear_model import LinearRegression

    # dataset = mlflow.data.from_pandas(
    #     pd.concat([X, y]),
    #     source='https://www.kaggle.com/datasets/guilherme26/house-pricing-in-belo-horizonte',
    #     name='House Pricing in Belo Horizonte',
    #     targets='price'
    # )

    categorical_features = variable_descriptions['categorical']
    numerical_features = variable_descriptions['numerical']
    target = variable_descriptions['target']

    X = df[categorical_features + numerical_features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=0.2, random_state=random_state
    )

    def dataframe_to_dict(df: pd.DataFrame):
        return df.to_dict('records')

    with mlflow.start_run(run_name='Simple Linear Regression'):
        tag_run()

        regressor = LinearRegression()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        df_transformer = FunctionTransformer(dataframe_to_dict)

        categorical_transformer = Pipeline(
            steps=[
                (
                    'to_dict',
                    df_transformer,
                ),  # returns a list of dicts
                (
                    'vectorizer',
                    DictVectorizer(),
                ),  # list of dicts -> feature matrix
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        pipeline = make_pipeline(
            preprocessor,
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

    return run_model.run_id, {}


def train_elastic_net_regression(
    df: pd.DataFrame,
    variable_descriptions: Dict[str, list[str]],
    random_state: int = 42,
    max_evals: int = 100,
) -> Pipeline:
    """Train a regression model on the given data usin the ElatictNet algorithm.

    Args:
        df (pandas.DataFrame): Input data.
        variable_descriptions (Dict[str, list[str]]): The description of the variables.
            It should contain the following keys: 'categorical', 'numerical' and 'target'.
        random_state: random state to use for reproducibility (default to 42).
        max_evals: maximum number of evaluations for the hyperparameter optimization
            (default to 100).

    # Returns:
    #     sklearn.pipeline.Pipeline: Trained linear regression model with DictVectorizer.
    """
    from sklearn.linear_model import ElasticNet

    def dataframe_to_dict(df: pd.DataFrame):
        return df.to_dict('records')

    categorical_features = variable_descriptions['categorical']
    numerical_features = variable_descriptions['numerical']
    target = variable_descriptions['target']

    X = df[categorical_features + numerical_features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.values, test_size=0.2, random_state=random_state
    )

    def create_pipeline(params):
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'numerical',
                    Pipeline(steps=[('scaler', StandardScaler())]),
                    numerical_features,
                ),
                (
                    'categorical',
                    Pipeline(
                        steps=[
                            (
                                'to_dict',
                                FunctionTransformer(dataframe_to_dict),
                            ),
                            ('vectorizer', DictVectorizer()),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        pipeline = make_pipeline(
            preprocessor,
            TransformedTargetRegressor(
                regressor=ElasticNet(**params),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        )

        return pipeline

    def objective(params):

        with mlflow.start_run(run_name='ElasticNet Optimization', nested=True):
            mlflow.log_params(params)

            pipeline = create_pipeline(params)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            rmse_optimized = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric('rmse', rmse_optimized)
            mlflow.sklearn.log_model(pipeline, 'model')

            return {'loss': rmse_optimized, 'status': STATUS_OK}

    search_space = {
        'alpha': hp.uniform('alpha', 0.0001, 0.001),
        'l1_ratio': hp.uniform('l1_ratio', 0, 0.2),
        'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 1)),
        'random_state': random_state,
    }

    with mlflow.start_run(run_name='Elasticnet'):
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(random_state),
        )

        # TODO: find if there is a better way to force a string value
        best_params['max_iter'] = int(best_params['max_iter'])

        logger.info('Best parameters are : %s', best_params)

        pipeline = create_pipeline(best_params)

        mlflow.log_params(best_params)

        best_model = mlflow.sklearn.log_model(
            pipeline.fit(X_train, y_train), 'model'
        )

        return best_model.run_id, best_params


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
