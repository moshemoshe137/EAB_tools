import itertools
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def base_path() -> Path:
    """Get the current folder of the test"""
    # Thanks to https://stackoverflow.com/a/70111624/9655521
    return Path(__file__).parent


iris_df: pd.DataFrame = pd.read_csv(Path(__file__).parent / "io/data/iris.csv")


@pytest.fixture
def iris() -> pd.DataFrame:
    """The iris dataset as a pandas DataFrame."""
    return iris_df


@pytest.fixture(params=iris_df.columns)
def iris_cols(iris: pd.DataFrame, request: pytest.FixtureRequest) -> pd.Series:
    """Return iris dataframe columns, one after the next"""
    return iris_df[request.param]


@pytest.fixture(
    params=[
        pytest.param(pd.Series([1, 2, 3] * 3, dtype="int32"), id='int32series'),
        pytest.param(pd.Series([None, 2.5, 3.5] * 3, dtype="float32"), id='float32series'),
        pytest.param(pd.Series(["a", "b", "c"] * 3, dtype="category"), id='category_series'),
        pytest.param(pd.Series(["d", "e", "f"] * 3), id='object_series'),
        pytest.param(pd.Series([True, False, True] * 3), id='bool_series'),
        pytest.param(pd.Series(pd.date_range("20130101", periods=9)), id='datetime_series'),
        pytest.param(pd.Series(pd.date_range("20130101", periods=9, tz="US/Eastern")), id='datetime_tz_series'),
        pytest.param(pd.Series(pd.timedelta_range("2000", periods=9)), id='timedelta_series'),
    ]
)
def series(request: pytest.FixtureRequest) -> pd.Series:
    """Return several series with unique dtypes"""
    # Fixture borrowed from pandas from
    # https://github.com/pandas-dev/pandas/blob/5b2fb093f6abd6f5022fe5459af8327c216c5808/pandas/tests/util/test_hashing.py
    return request.param


pairs = list(itertools.permutations(iris_df.columns, 2))
@pytest.fixture(params=pairs, ids=map(str, pairs))
def multiindex(
        iris: pd.DataFrame,
        request: pytest.FixtureRequest
) -> pd.MultiIndex:
    """Return MultiIndexes created from pairs of iris cols"""
    a_col, b_col = request.param
    a, b = iris[a_col], iris[b_col]
    return pd.MultiIndex.from_arrays([a, b])

