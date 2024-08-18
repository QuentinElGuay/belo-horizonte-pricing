import json
import logging
from os import getenv

from library.serve import predict
from mlflow import set_tracking_uri


def lambda_handler(event, context):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    TRACKING_SERVER_URI = getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    set_tracking_uri(TRACKING_SERVER_URI)

    # TODO: those should be deduced from the event dictionnary
    MODEL_NAME = getenv('MODEL_NAME', 'belo_horizonte_price_regression')

    inputs = json.loads(event['body'])
    prediction = predict(MODEL_NAME, inputs)
    price = prediction[0]

    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_price': price}),
    }

    return response
