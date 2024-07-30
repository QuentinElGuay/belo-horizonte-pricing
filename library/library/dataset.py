import logging
from math import ceil
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def read_csv(file_path: str) -> pd.DataFrame:
    """Read dataset from a CSV file and return a DataFrame containing the data.

    Args:
        file_path (str): Path to the csv file.

    Returns:
        pandas.DataFrame: DataFrame containing the data from the csv file
    """
    logger.info('Reading file %s', file_path)

    df = pd.read_csv(
        file_path,
        encoding='utf-8',
        dtype={'rooms': str, 'square-foot': str, 'garage-places': str},
        skipinitialspace=True,
    )

    logger.info(f'The dataset %s contains %s rows.', file_path, len(df))
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names by converting them to snake case.

    Args:
        df (pandas.DataFrame): DataFrame to standardize column names.

    Returns:
        pandas.DataFrame: DataFrame with standardized column names.
    """
    logger.info('Standardizing the column names...')
    df.rename(
        columns={col: col.replace('-', '_').lower() for col in df.columns},
        inplace=True,
    )

    return df


def convert_numeric_columns(
    df: pd.DataFrame, numeric_columns: Tuple[str]
) -> pd.DataFrame:
    """Convert columns of a DataFrame to numeric type.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        numeric_columns (Tuple[str]): List of column names to convert to numeric type.

    Returns:
        pandas.DataFrame: DataFrame with numeric columns cleaned.
    """
    logger.info('Converting the numeric variables received as text...')

    df.adm_fees = df.adm_fees.fillna(0)

    for column in numeric_columns:
        df[column] = df[column].replace('--', '0')
        df[column] = df[column].apply(
            lambda v: str(
                ceil((int(v.split('-')[0]) + int(v.split('-')[1])) / 2)
                if '-' in v
                else v
            )
        )
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers based on the price and price per square-foot to remove the luxury
     properties from the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to file.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    logger.info('Removing outliers...')
    original_length = len(df)

    # TODO: Those values should be defined by a configuration file (see hydra)
    MIN_ROOMS = 1
    MAX_ROOMS = 6
    MIN_GARAGE_PLACES = 0
    MAX_GARAGE_PLACES = 6
    MIN_AREA = 10
    MAX_AREA = 1000
    MIN_PRICE = 1e5
    MAX_PRICE = 4e6

    df = df[
        df.rooms.between(MIN_ROOMS, MAX_ROOMS)
        & df.garage_places.between(MIN_GARAGE_PLACES, MAX_GARAGE_PLACES)
        & df.square_foot.between(MIN_AREA, MAX_AREA)
        & df.price.between(MIN_PRICE, MAX_PRICE)
    ]

    # # TODO: check if really usefull
    # df['square_foot_price'] = df.price / df.square_foot
    # df = df[
    #     df.square_foot_price.between(
    #         df.square_foot_price.quantile(0.01),
    #         df.square_foot_price.quantile(0.99),
    #     )
    # ]

    logger.info(
        f'  Removed %s rows containing outliers.',
        original_length - len(df),
    )

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset for the training process.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    """
    logger.info('Cleaning the dataset:')
    original_length = len(df)

    logger.info('Dropping duplicates...')
    df = df.drop_duplicates()
    logger.info('   Dropped %s duplicated rows.', len(df) - original_length)

    df = standardize_column_names(df)

    df = convert_numeric_columns(df, ('rooms', 'square_foot', 'garage_places'))

    df = remove_outliers(df)

    # TODO: check if really usefull
    categorical = ['neighborhood']
    df[categorical] = df[categorical].astype(str)

    df.reset_index(drop=True, inplace=True)

    logger.info(f'Number of rows after cleaning the dataset: %s', len(df))

    return df


def get_dataset(file_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read and clean the dataset file for the training process.

    Args:
        file_path (str): Path to the csv file..

    Returns:
        Tuple[pd.DataFrame, Dict[str,str]]: A cleaned DataFrame and a description of
        the variables.
    """
    df = read_csv(file_path)
    df = clean_dataset(df)

    variables = {
        'target': 'price',
        'categorical': ['neighborhood'],
        'numerical': ['square_foot', 'garage_places', 'rooms'],
    }

    return df, variables


def split_test_datase(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame]:
    """Split the dataset into training and test datasets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The percentage of the dataset to include in the test split.
         Must be between 0 and 1.
        random_state (int): The random state to use for the split for reproducibility.

    Returns:
        Tuple[pd.DataFrame]: The training and test datasets.
    """
    logger.info('Splitting the dataset')
    X_train, X_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    logger.info(
        '   Split the dataset into datasets of %s and %s rows.',
        len(X_train),
        len(X_test),
    )

    return X_train, X_test


def prepare_features(X: Dict[str, Any] | pd.DataFrame) -> Dict[str, Any]:
    """Prepare the features to be used by the model.

    Args:
        X (Dict[str, Any] | pd.DataFrame): The feature to prepare.

    Returns:
        numpy.ndarray: The prepared features.
    """
    # TODO: this should be a parameter
    NUMERIC_COLUMNS = ('rooms', 'square_foot', 'garage_places')

    if isinstance(X, dict):
        df = pd.DataFrame.from_records([X])
    elif isinstance(X, pd.DataFrame):
        df = X
    else:
        raise ValueError(
            'X must be a feature dictionary or a pandas DataFrame and not a %s.',
            type(X),
        )

    df = convert_numeric_columns(df, NUMERIC_COLUMNS)

    return df.to_dict(orient='records')
