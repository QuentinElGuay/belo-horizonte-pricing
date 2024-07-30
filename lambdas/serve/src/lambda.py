
import json
import logging
from os import getenv

from mlflow import set_tracking_uri

# from belo_horizonte_estate_pricing.library.serve import predict


def lambda_handler(event, context):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    TRACKING_SERVER_URI = getenv('TRACKING_SERVER_URI', 'http://localhost:5000')
    set_tracking_uri(TRACKING_SERVER_URI)

    # TODO: those should be deduced from the event dictionnary
    model_name = 'belo_horizonte_estate_pricing'
    inputs = {
        'adm_fees': '',
        'neighborhood': 'Miramar',
        'square_foot': '79',
        'rooms': '2',
        'garage_places': '--',
    }

    price = 1000

    # prediction = predict(model_name, inputs)
    # price = prediction[0]

    response =  {
        'statusCode': 200,
        'body': json.dumps({
            'predicted_price': price
        }),
    }

    return response
