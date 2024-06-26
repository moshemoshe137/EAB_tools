"""Tests for hashing functionality in the EAB_tools package.

This module contains tests to verify the consistent behavior and functionality of
hashing utilities applied to different data structures such as `pandas` DataFrames,
`matplotlib` figures, and more.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import EAB_tools._testing as tm
from EAB_tools.util.hashing import (
    hash_df,
    hash_mpl_fig,
)


class TestHashDf:
    """
    Test suite for the `hash_df` function from `EAB_tools.util.hashing`.

    This class includes tests that verify hashing consistency and uniqueness based on
    DataFrame content changes, including different aspects like DataFrame structure,
    styling, and handling of different data types like Series and MultiIndex.

    Methods
    -------
    test_basic_consistency:
        Verify that hashing a DataFrame multiple times results in the same hash.
    test_expected_iris:
        Confirm that the hash of a known DataFrame matches an expected hash.
    test_expected_iris_columns:
        Ensure that hashing individual DataFrame columns yields expected results.
    test_expected_iris_index:
        Test that converting DataFrame columns to an Index and hashing them gives
        expected hashes.
    test_series_hash_consistency:
        Check hashing consistency on a `pandas` Series.
    test_columns_affect_hash:
        Confirm that changing DataFrame column names affects the resulting hash.
    test_index_hash_consistency:
        Verify that hashes are consistent when hashing a `pandas` Index multiple times.
    test_multiindex_hash_consistency:
        Check hashing consistency for `pandas` MultiIndex objects.
    test_styler_consistency:
        Ensure that styling a DataFrame does not affect its hash when the style is
        applied consistently.
    test_styler_affects_hash:
        Verify that different styles result in different hashes for the same DataFrame.
    """

    def test_basic_consistency(self, iris: pd.DataFrame) -> None:
        """
        Verify that the hash of a DataFrame is consistent across multiple calls.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame to be hashed. The Iris DataFrame.
        """
        assert hash_df(iris) == hash_df(iris)

    def test_expected_iris(self, iris: pd.DataFrame) -> None:
        """
        Check that the hash of a known DataFrame matches an expected hash.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame to be hashed. The Iris DataFrame.
        """
        expected = "d7565"
        result = hash_df(df=iris, max_len=5, usedforsecurity=False)
        assert result == expected

    def test_expected_iris_columns(self, iris: pd.DataFrame) -> None:
        """
        Test that hashing individual DataFrame columns produces expected hashes.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame whose columns are to be hashed individually.  The Iris
            DataFrame.
        """
        hash_df_kwargs: dict[str, Any] = {"max_len": 5, "usedforsecurity": False}
        expected = ["3ce8a", "7e266", "9759d", "2a060", "ea451"]
        result = [hash_df(iris[column], **hash_df_kwargs) for column in iris]
        assert expected == result

    def test_expected_iris_index(self, iris: pd.DataFrame) -> None:
        """
        Ensure hashing the Index of a DataFrame gives expected results.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame whose column indices are to be hashed.  The Iris DataFrame.
        """
        hash_df_kwargs: dict[str, Any] = {"max_len": 5, "usedforsecurity": False}
        expected = ["4f53f", "5da34", "ca488", "f1607", "5517a"]
        result = [hash_df(pd.Index(iris[column]), **hash_df_kwargs) for column in iris]

        assert expected == result

    def test_series_hash_consistency(self, series: pd.Series) -> None:
        """
        Verify that the hash of a `pandas` Series is consistent across multiple calls.

        Parameters
        ----------
        series : pd.Series
            The Series to be hashed.
        """
        a = hash_df(series)
        b = hash_df(series)
        assert a == b

    def test_columns_affect_hash(self, series: pd.Series) -> None:
        """
        Confirm that changing column names in a DataFrame results in new hashes.

        Parameters
        ----------
        series : pd.Series
            The Series to be converted into a DataFrame and hashed.
        """
        a = pd.DataFrame({"foo": series})
        b = pd.DataFrame({"bar": series})

        assert hash_df(a) != hash_df(b)

    def test_index_hash_consistency(self, series: pd.Series) -> None:
        """
        Test that hashing a `pandas` Index multiple times results in consistent hashes.

        Parameters
        ----------
        series : pd.Series
            The Series whose index is to be hashed.
        """
        index = pd.Index(series)
        assert hash_df(index) == hash_df(index)

    def test_multiindex_hash_consistency(self, multiindex: pd.MultiIndex) -> None:
        """
        Verify that hashing a `pandas` MultiIndex is consistent across multiple calls.

        Parameters
        ----------
        multiindex : pd.MultiIndex
            The MultiIndex to be hashed.
        """
        assert hash_df(multiindex) == hash_df(multiindex)

    def test_styler_consistency(self, iris: pd.DataFrame) -> None:
        """
        Check that a DataFrame with consistent styling produces the same hash.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame to be styled and hashed. The Iris DataFrame.
        """
        iris_style = iris.style.highlight_min().highlight_max().bar()
        assert hash_df(iris, iris_style) == hash_df(iris, iris_style)

    def test_styler_affects_hash(self, iris: pd.DataFrame) -> None:
        """
        Verify that different styling of the same DataFrame results in different hashes.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame to be styled and hashed in different ways.
        """
        style1 = iris.style.highlight_min()
        style2 = iris.style.highlight_max()
        assert hash_df(iris, style1) != hash_df(iris, style2)


@pytest.mark.flaky(rerun_filter=tm._is_tkinter_error, max_runs=5)
class TestHashMPLfig:
    """
    Test suite for the `hash_mpl_fig` function from `EAB_tools.util.hashing`.

    This class tests the hashing functionality specifically designed for `matplotlib`
    figures to ensure that hashing is consistent and sensitive to changes in the
    figure's content.

    Methods
    -------
    test_expected_hash:
        Verify that the hash of a `matplotlib` figure matches an expected hash.
    test_basic_consistency:
        Confirm that the hash of a figure is consistent across multiple calls.
    test_sensitivity:
        Test that changes in a figure's content lead to changes in its hash.
    """

    def test_expected_hash(self) -> None:
        """
        Check that the hash of a `matplotlib` figure matches an expected hash.

        This test creates a plot with specific data and verifies that the hash of the
        figure is as expected, demonstrating the hash function's ability to generate
        consistent and reproducible hashes for figures.
        """
        expected = "c89d743"

        # Make up some data
        x = np.linspace(0, 4 * np.pi, 10**6)
        y = np.sin(x)

        # Plot made up data
        fig, ax = plt.subplots()
        ax.plot(x, y)

        kwargs: dict[str, Any] = {"max_len": 7, "usedforsecurity": False}
        assert hash_mpl_fig(fig, **kwargs) == expected

    def test_basic_consistency(self, mpl_figs_and_axes: plt.Figure | plt.Axes) -> None:
        """
        Ensuring hashing `matplotlib` objects is consistent.

        Parameters
        ----------
        mpl_figs_and_axes : plt.Figure | plt.Axes
            The `matplotlib` object to be hashed.
        """
        a = hash_mpl_fig(mpl_figs_and_axes)
        b = hash_mpl_fig(mpl_figs_and_axes)
        assert a == b

    def test_sensitivity(self, mpl_axes: plt.Axes, seed: int = 0) -> None:
        """
        Test the sensitivity of the hashing function to changes in a figure's content.

        This test modifies a `matplotlib` axes by adding a randomly placed scatter plot
        and verifies that this change affects the hash of the axes.

        Parameters
        ----------
        mpl_axes : plt.Axes
            The `matplotlib` axes to be tested.
        seed : int, optional
            The seed for the random number generator to ensure reproducibility.
        """
        a = hash_mpl_fig(mpl_axes)

        rng = np.random.default_rng(seed=seed)
        xy = rng.random(2)
        color = rng.random(3)
        mpl_axes.scatter(xy[0], xy[1], color=color)
        b = hash_mpl_fig(mpl_axes)
        assert a != b
