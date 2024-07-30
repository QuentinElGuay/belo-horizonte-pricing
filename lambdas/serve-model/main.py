import logging
from os import getenv
from typing import Any, Dict

import mlflow

from library.serve import predict


def main(model_name: str, inputs: Dict[str, Any]):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    TRACKING_SERVER_URI = getenv(
        'TRACKING_SERVER_URI', 'http://localhost:5000'
    )
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)

    prediction = predict(model_name, inputs)

    logger.info('Prediction: %s', prediction)


if __name__ == '__main__':

    model_name = 'belo_horizonte_estate_pricing'
    inputs = {
        'adm_fees': '',
        'neighborhood': 'Miramar',
        'square_foot': '79',
        'rooms': '2',
        'garage_places': '--',
    }

    main(model_name, inputs)
