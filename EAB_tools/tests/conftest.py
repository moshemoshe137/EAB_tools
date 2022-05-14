import itertools
from pathlib import Path
from typing import (
    Callable,
    Union,
    Literal
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


@pytest.fixture(params=[
    np.sin,  # any -> float
    pytest.param(lambda arr: np.exp(-arr), id='exp(-x)'),  # any -> float
    pytest.param(lambda x: x ** 2, id='lambda squared'),  # int -> int and float -> float
    pytest.param(lambda arr: np.rint(arr).astype(int), id='rint')  # any -> int
], name='func')
def plot_func(request: pytest.FixtureRequest) -> Callable:
    """A variety of funcs callable on numeric ndarrays"""
    return request.param


@pytest.fixture(params=[
    np.linspace(0, 10 ** -5, dtype=float),
    np.linspace(0, 499, num=500, dtype='int32'),
    np.linspace(0, 2**33, 2**10 + 1, dtype='int64')
], ids=lambda arr: str(arr.dtype))
def x_values(request: pytest.FixtureRequest) -> np.ndarray:
    """func inputs of different dtypes"""
    return request.param


@pytest.fixture(params=['fig', 'ax'])
def _fig_or_ax(request: pytest.FixtureRequest) -> Literal['fig', 'ax']:
    """Either returns 'fig' or 'ax'"""
    return request.param


@pytest.fixture
def mpl_plots(
        func: Callable[[np.ndarray], np.ndarray],
        x_values: np.ndarray
) -> tuple[plt.Figure, plt.Axes]:
    """Returns dict of {fix, ax}, for various funcs and domains"""
    x = x_values
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)

    yield dict(fig=fig, ax=ax)
    plt.close(fig)


@pytest.fixture
def mpl_axes(mpl_plots: tuple[plt.Figure, plt.Axes]) -> plt.Axes:
    """Returns a variety of `plt.Axes` objects"""
    return mpl_plots['ax']


@pytest.fixture
def mpl_figs(mpl_plots: tuple[plt.Figure, plt.Axes]) -> plt.Figure:
    """Returns a variety of `plt.Figure` objects"""
    return mpl_plots['fig']


@pytest.fixture
def mpl_figs_and_axes(
        mpl_plots: tuple[plt.Figure, plt.Axes],
        _fig_or_ax: Literal['fig', 'ax']
) -> Union[plt.Figure, plt.Axes]:
    """Returns either the figure or the axis of various plots"""
    return mpl_plots[_fig_or_ax]
