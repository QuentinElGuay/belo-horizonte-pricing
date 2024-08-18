import pytest
from pandas.api.types import is_numeric_dtype

from src import dataset


@pytest.fixture
def test_dataset_path():
    yield 'tests/fixtures/test_dataset.csv'


def test_read_csv(test_dataset_path):
    """Check the DataFrame created by the read_csv function
    Args:
        test_dataset_path (str): path to the test dataset
    """
    df = dataset.read_csv(test_dataset_path)

    assert len(df) == 15
    assert set(df.columns) == set(
        (
            'address',
            'adm-fees',
            'garage-places',
            'price',
            'rooms',
            'square-foot',
            'neighborhood',
            'city',
            'latitude',
            'longitude',
        )
    )

    text_columns = (
        'address',
        'neighborhood',
        'city',
        'garage-places',
        'rooms',
        'square-foot',
    )
    for column in text_columns:
        assert df[column].dtype == 'object'

    float_columns = ('adm-fees', 'price', 'latitude', 'longitude')
    for column in float_columns:
        assert df[column].dtype == 'float64'


def test_standardize_column_names(test_dataset_path):
    """Test if the tandardize_column_names function rename correctly the columns of
    the dataset. Column names must be:
    - lowercase
    - with underscores instead of hyphens.

    Args:
        test_dataset_path (str): : path to the test dataset
    """
    df = dataset.read_csv(test_dataset_path)
    assert set(('adm-fees', 'garage-places', 'square-foot')).issubset(
        set(df.columns)
    )

    df = dataset.standardize_column_names(df)
    assert set(('adm_fees', 'garage_places', 'square_foot')).issubset(
        set(df.columns)
    )
    for column in set(df.columns):
        assert '-' not in column
        assert column.islower()


def test_convert_numeric_columns(test_dataset_path):
    """Test if the convert_numeric_columns function converts correctly the numeric
    variables of the dataset.

    Args:
        test_dataset_path (str): : path to the test dataset
    """
    df = dataset.read_csv(test_dataset_path)
    df = dataset.standardize_column_names(df)

    columns_to_convert = ('garage_places', 'rooms', 'square_foot')
    df = dataset.convert_numeric_columns(df, columns_to_convert)

    for column in columns_to_convert:
        assert is_numeric_dtype(df[column])

    # Empty admin-fees are replaced by 0
    assert df.iloc[0]['adm_fees'] == 0
    # range values for garage_places and rooms are replaced by the upper mean value of the range
    assert df[df['address'] == 'Range rooms - OK'].iloc[0]['rooms'] == 3
    assert (
        df[df['address'] == 'Range garage-places - OK'].iloc[0][
            'garage_places'
        ]
        == 3
    )
    assert (
        df[df['address'] == 'Range square-foot - OK'].iloc[0]['square_foot']
        == 110
    )


def test_remove_outliers(test_dataset_path):
    """Test if the remove_outliers function correctly the rows containing outliers
     values from the dataset.

    Args:
        test_dataset_path (str): : path to the test dataset
    """
    df = dataset.read_csv(test_dataset_path)
    df = dataset.standardize_column_names(df)
    df = dataset.convert_numeric_columns(
        df, ('garage_places', 'rooms', 'square_foot')
    )
    df = dataset.remove_outliers(df)

    # Properties without price or outside the price range are removed
    assert len(df[df['address'] == 'Without price - KO']) == 0
    assert len(df[df['address'] == 'Over maximum price - KO']) == 0
    assert len(df[df['address'] == 'Under minimum price - KO']) == 0
    # properties without room or over the rooms range are removed
    assert len(df[df['address'] == 'Without rooms - KO']) == 0
    assert len(df[df['address'] == 'Over maximum rooms - KO']) == 0
    # properties over the garage_places range are removed
    assert len(df[df['address'] == 'Over maximum garage-places - KO']) == 0
    # properties without square_foot or outside the square_foot range are removed
    assert len(df[df['address'] == 'Without square-foot - KO']) == 0
    assert len(df[df['address'] == 'Under minimum square-foot - KO']) == 0
    assert len(df[df['address'] == 'Over maximum square-foot - KO']) == 0

    # 6 rows are not filtered
    assert len(df) == 6


def test_clean_dataset(test_dataset_path):
    """Test if the clean_dataset function correctly cleans the dataset.

    Args:
        test_dataset_path (str): : path to the test dataset
    """
    df = dataset.read_csv(test_dataset_path)
    df = dataset.clean_dataset(df)

    assert len(df) == 6
