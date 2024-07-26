import logging
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from typing import List, Tuple

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
        encoding="utf-8",
        dtype={'rooms': str, 'square-foot': str, 'garage-places': str}
    )

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names by converting them to snake case.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: DataFrame with standardized column names.
    """
    df.rename(
        columns={col: col.replace('-', '_').lower() for col in df.columns},
        inplace=True
    )

    return df


def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert columns to numeric type.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: DataFrame with numeric columns cleaned.
    """
    for column in columns:
        df[df[column].str.contains('-')][column].apply(
            lambda v: str(round(int(v.split('-')[1]) / 2))
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset for the training process.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    """
    logger.info('Cleaning dataset')

    df = standardize_column_names(df)
    df = df.drop_duplicates()

    df['rooms'] = df['rooms'].replace('--', np.nan)
    df['square_foot'] = df['square_foot'].replace('--', np.nan)
    df['garage_places'] = df['garage_places'].replace('--', '0')
    df.adm_fees = df.adm_fees.fillna(0)
    df.dropna(axis=0, inplace=True)

    convert_numeric_columns(df, ['rooms', 'square_foot', 'garage_places'])
    df.dropna(axis=0, inplace=True)

    df.reset_index(drop=True, inplace=True)

        # filter outliers
    df = df[df.price.between(1e5, 4e6)]
    df['square_foot_price'] = (df.price / df.square_foot)
    df = df[df.square_foot_price.between(
        df.square_foot_price.quantile(0.01),
        df.square_foot_price.quantile(0.99))
    ]

    # df['rooms'] = df.rooms.apply(lambda x: int(x) if x < 5 else 5)
    # df['garage_places'] = df.garage_places.apply(lambda x: int(x) if x < 5 else 5)
    df = df[(df.rooms <= 6) & (df.garage_places <= 6) & (df.square_foot <= 1000)]

    scaler = StandardScaler()
    df['square_foot'] = scaler.fit_transform(df[['square_foot']])

    return df


def split_test_datase(df: pd.DataFrame, test_size:float, random_state:int) -> Tuple[pd.DataFrame]:
    """Split the dataset into training and test datasets.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df, test_size=0.2, random_state=random_state)
    
    return pd.concat(X_train, y_train), pd.concat(X_test, y_test)

def get_dataset(file_path: str) -> pd.DataFrame:
    """Read and clean the dataset file for the training process.

    Args:
        file_path (str): Path to the csv file..

    Returns:
        pandas.DataFrame: A cleaned DataFrame.
    """
    df = read_csv(file_path)
    df = clean_dataset(df)

    return df
