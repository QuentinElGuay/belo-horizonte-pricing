import json
import logging
import os

import mlflow
from library.serve import predict


def lambda_handler(event, context):
    """Handler to be called by a lambda function

    Args:
        event (dict): A dictionnary containing the request parameters.
        context (dict): A dicitonnary containing the execution context.

    Returns:
        dict: A dictionnary containing the response to the request.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('Starting prediction')
    logger.debug('Event: %s', event)

    TRACKING_SERVER_URI = os.getenv(
        'TRACKING_SERVER_URI', 'http://mlflow_ui:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    # TODO: those should be deduced from the event dictionnary
    MODEL_NAME = os.getenv('MODEL_NAME', 'belo-horizonte-estate-pricing')

    inputs = json.loads(event['body'])
    prediction = predict(MODEL_NAME, inputs)
    price = prediction[0]

    logger.info('Prediction: %s', price)

    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_price': price}),
    }

    return response
