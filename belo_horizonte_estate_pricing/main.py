from os import getenv
import logging
from matplotlib import pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import  root_mean_squared_error
from sklearn.model_selection import train_test_split

from library.dataset import get_dataset
from library.train import train_simple_linear_regression  ##, train_regularized_regression

logger = logging.getLogger(__name__)

TRACKING_SERVER_URI = getenv('TRACKING_SERVER_URI', 'http://localhost:5000')
RANDOM_STATE = 42

def main(experiment_name:str, dataset_path:str):

    logging.basicConfig(level=logging.INFO)

    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    mlflow.set_experiment(experiment_name)

    np.random.seed(RANDOM_STATE)

    # Get dataset
    df = get_dataset(dataset_path)

    variables = {
        'target': 'price',
        'categorical': ['neighborhood'],
        'numerical': ['square_foot', 'garage_places', 'rooms']
    }

    # Split the original dataset in training and test datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        df[variables['categorical'] + variables['numerical']],
        df[variables['target']],
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    pipeline = train_simple_linear_regression(X_train, y_train)
    y_pred = pipeline.predict(X_test.to_dict('records'))

    print(root_mean_squared_error(y_test, y_pred))
    print(pipeline.score(X_test.to_dict('records'), y_test))

    # df_test = read_csv('data/data_kaggle_2021.csv')
    # df_test = prepare_dataset(df_test)

    # X_test = df_test[categorical + numerical]
    # y_test = df_test[target]

    # y_pred = pipeline.predict(X_test.to_dict('records'))
    # print(root_mean_squared_error(y_test, y_pred))
    # print(pipeline.score(X_test.to_dict('records'), y_test))

    # print(X_test.to_dict('records')[0], np.expm1(y_test.iloc[0]), np.expm1(y_pred[0]))

    # train_regularized_regression(X_train, y_train)

    # x = []
    # plt.plot(y_pred, linestyle = 'dotted')
    # for i in range(0, len(y_test)):
    #     x.append(i)
    # plt.scatter(x, y_test)
    # plt.show()



if __name__ == "__main__":
    
   main('belo-horizonte-test-kaggle_2021', 'data/data_kaggle_2021.csv')
