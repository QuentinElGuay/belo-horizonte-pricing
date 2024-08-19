import json
import logging
import os

import mlflow
from library.serve import predict


def lambda_handler(event, context):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    TRACKING_SERVER_URI = os.getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    # TODO: those should be deduced from the event dictionnary
    MODEL_NAME = os.getenv('MODEL_NAME', 'belo-horizonte-estate-pricing')

    inputs = json.loads(event['body'])
    prediction = predict(MODEL_NAME, inputs)
    price = prediction[0]

    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_price': price}),
    }

    return response
