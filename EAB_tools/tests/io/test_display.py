# pylint: disable=C0114, C0116
"""Test suite for verifying functionality data display and saving capabilities."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
from pathlib import Path
import re
from typing import (
    Any,
    ContextManager,
)

from matplotlib import pyplot as plt
import pandas as pd
import pytest

import EAB_tools as eab
from EAB_tools import (
    display_and_save_df,
    display_and_save_fig,
    sanitize_filename,
)
import EAB_tools._testing as tm

try:
    import openpyxl as _openpyxl  # noqa: F401 # 'openpyxl ... imported but unused

    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False


SaveImageTrueParam = pytest.param(True, marks=pytest.mark.slow)
SaveExcelTrueParam = pytest.param(
    True, marks=pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl required")
)


@pytest.mark.parametrize(
    "save_image",
    [SaveImageTrueParam, False],
    ids="save_image={}".format,
)
@pytest.mark.parametrize(
    "save_excel",
    [SaveExcelTrueParam, False],
    ids="save_excel={}".format,
)
class TestDisplayAndSaveDf:
    """
    Tests for `display_and_save_df` function.

    These tests validate the `display_and_save_df` function's ability to accurately
    handle and render `pandas` DataFrames and Series, ensuring correct display,
    appropriate formatting, and reliable saving under various conditions. Tests cover a
    range of data types, including numeric, string, and datetime, and examine both image
    and Excel file outputs where applicable.

    Attributes
    ----------
    save_image : bool
        Controls whether plots are saved as images during tests.
    save_excel : bool
        Determines if DataFrames are saved as Excel files, dependent on the presence
        of `openpyxl`.

    Methods
    -------
    test_doesnt_fail(iris, save_image, save_excel):
        Verifies that no exceptions are raised for typical DataFrame inputs.
    test_series_doesnt_fail(series, save_image, save_excel):
        Checks that `pandas` Series are processed correctly without errors.
    test_multiindex_index(iris, save_image, save_excel):
        Ensures correct handling of DataFrames with MultiIndexes.
    col_name_from_iris_single_col_subset(col_name):
        Utility function to extract a column name from a string or a `pandas` Index.
    test_styler_expected_text(
        iris, iris_single_col_subset, kwargs, save_image, save_excel
    ):
        Confirms that DataFrame Styler objects render with the specified text formats.
    test_auto_percentage_format(iris, iris_cols, save_image, save_excel):
        Tests automatic application of percentage formatting based on column names.
    test_auto_thousands_format(iris, iris_cols, save_image, save_excel):
        Verifies correct application of thousands separators in appropriate contexts.
    test_auto_float_format(iris, iris_cols, precision, save_image, save_excel):
        Assesses precision handling in floating-point formatting.
    test_datetime_format(datetime_df, subset, strftime, save_image, save_excel):
        Checks correct date formatting across various strftime configurations.
    test_auto_datetime_format(datetime_and_float_df, strftime, save_image, save_excel):
        Evaluates the effectiveness of automatic date formatting.
    test_hide_index(iris, save_image, save_excel):
        Tests the ability to toggle visibility of the DataFrame index in outputs.
    test_ryg_background_gradient(iris, iris_single_col_subset, save_image, save_excel):
        Tests the application of a red-yellow-green gradient background based on values.
    test_ryg_background_vmin(iris, iris_single_col_subset, save_image, save_excel):
        Assesses the impact of setting a minimum value for the RYG gradient background.
    test_ryg_background_vmax(iris, iris_single_col_subset, save_image, save_excel):
        Tests the effect of setting a maximum value for the RYG gradient background.
    test_gyr_background_vmin(iris, iris_single_col_subset, save_image, save_excel):
        Evaluates the GYR gradient's response to a set minimum value.
    test_gyr_background_vmax(iris, iris_single_col_subset, save_image, save_excel):
        Validates the effects of setting a maximum value for GYR background styling.
    test_bar_style(iris, iris_single_col_subset, save_image, save_excel):
        Tests bar style formatting for visual data representation in DataFrame cells.
    bar_pcnt_from_html(html):
        Utility to extract bar percentage values from the HTML of a styled DataFrame.
    test_bar_vmin(iris, iris_single_col_subset, save_image, save_excel):
        Tests the impact of setting a minimum value for bar styles.
    test_bar_vmax(iris, iris_single_col_subset, save_image, save_excel):
        Evaluates the effect of setting a maximum value for bar styles.
    """

    def test_doesnt_fail(
        self, iris: pd.DataFrame, save_image: bool, save_excel: bool
    ) -> None:
        """
        Test that `display_and_save_df` executes without errors when using a DataFrame.

        This test ensures that the `display_and_save_df` function can process a typical
        DataFrame and complete the operation without throwing exceptions, including the
        options to save outputs as images and Excel files if specified.

        Parameters
        ----------
        iris : pd.DataFrame
            A DataFrame containing the Iris dataset.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        display_and_save_df(iris, save_image=save_image, save_excel=save_excel)

    def test_series_doesnt_fail(
        self, series: pd.Series, save_image: bool, save_excel: bool
    ) -> None:
        """
        Ensure `display_and_save_df` handles a `pandas` Series without failing.

        Similar to testing with DataFrames, this method checks that the
        `display_and_save_df` function can handle a `pandas` Series and perform saving
        actions without encountering errors.

        Parameters
        ----------
        series : pd.Series
            A pandas Series to be tested.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        display_and_save_df(series, save_image=save_image, save_excel=save_excel)

    def test_multiindex_index(
        self, iris: pd.DataFrame, save_image: bool, save_excel: bool
    ) -> None:
        """
        Verify `display_and_save_df` with a DataFrame having a MultiIndex.

        Tests the `display_and_save_df` function's ability to handle DataFrames with
        MultiIndexes both as rows and columns, ensuring functionality is maintained
        across different DataFrame configurations.

        Parameters
        ----------
        iris : pd.DataFrame
            A DataFrame with the Iris dataset, modified to include a MultiIndex.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        iris_mi = iris.set_index("Name", append=True)
        display_and_save_df(iris_mi, save_image=save_image, save_excel=save_excel)
        display_and_save_df(iris_mi.T, save_image=save_image, save_excel=save_excel)

    @staticmethod
    def col_name_from_iris_single_col_subset(col_name: str | pd.Index) -> str:
        """
        Extract the column name from an input that may be a string or a pandas Index.

        This utility function is designed to standardize the extraction of a single
        column name from either a direct string or a `pandas` Index object, facilitating
        the flexible handling of column identifiers within tests. If passed an Index,
        the first item in the Index will be returned.

        Parameters
        ----------
        col_name : str | pd.Index
            The input that may be a string or a `pandas` Index.

        Returns
        -------
        str
            The extracted column name as a string.
        """
        return col_name if isinstance(col_name, str) else col_name[0]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {
                    "format": "{:,}",
                    "display_and_save_kw": "thousands_format_subset",
                    "need_large_numbers": True,
                },
                id="thousands subset",
            ),
            pytest.param(
                {
                    "format": "{:.1%}",
                    "display_and_save_kw": "percentage_format_subset",
                    "percentage_format_precision": 1,
                },
                id="percentage subset",
            ),
            pytest.param(
                {
                    "format": "{:.0f}",
                    "display_and_save_kw": "float_format_subset",
                    "float_format_precision": 0,
                },
                id="float.0f",
            ),
            pytest.param(
                {
                    "format": "{:.1f}",
                    "display_and_save_kw": "float_format_subset",
                    "float_format_precision": 1,
                },
                id="float.1f",
            ),
            pytest.param(
                {
                    "format": "{:.2f}",
                    "display_and_save_kw": "float_format_subset",
                    "float_format_precision": 2,
                },
                id="float.2f",
            ),
        ],
    )
    def test_styler_expected_text(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        kwargs: dict[str, Any],
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Confirm that DataFrame Styler objects render with expected text formatting.

        This method tests specific formatting options applied via the
        `display_and_save_df` function to ensure that the resultant Styler objects
        contain text formatted according to the parameters provided in `kwargs`. This
        includes testing various numerical, percentage, and thousands formatting
        options.

        Parameters
        ----------
        iris : pd.DataFrame
            DataFrame containing the Iris dataset to be formatted.
        iris_single_col_subset : str | pd.Index
            The column used for specific format testing.
        kwargs : dict[str, Any]
            Additional keyword arguments specifying the format details.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        iris, kwargs = iris.copy(), kwargs.copy()
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if kwargs.pop("need_large_numbers", False):
            # Need large numbers to test thousands separator
            large_iris = iris.copy()
            numeric_cols = large_iris.select_dtypes("number").columns
            large_iris[numeric_cols] = (large_iris[numeric_cols] * 10**6).astype(int)
            iris = large_iris

        col = iris[col_name]
        an_expected_value = col.iloc[0]
        try:
            an_expected_value = kwargs.pop("format").format(an_expected_value)
        except ValueError:
            # Number formatting can't be applied to str
            # Just make sure everything goes ok
            an_expected_value = ""

        if pd.api.types.is_string_dtype(col):
            context: ContextManager[object | None] = pytest.raises(ValueError)
        else:
            context = tm.does_not_raise()
        with context:
            # str columns with incorrect format code should
            # throw a ValueError

            # d is really dict[str, Union[str, pd.Index]], but mypy complains
            d: dict[str, Any] = {
                kwargs.pop("display_and_save_kw"): iris_single_col_subset
            }
            styler = display_and_save_df(
                iris, save_image=save_image, save_excel=save_excel, **d, **kwargs
            )
            styler.to_html()
            html = styler.to_html()
            assert an_expected_value in html

    def test_auto_percentage_format(
        self,
        iris: pd.DataFrame,
        iris_cols: pd.Series,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Validate automatic percentage formatting based on DataFrame column name.

        This test ensures that percentage formatting is automatically applied to columns
        with names suggesting percentage data, assessing the correct visual rendering in
        outputs.

        Parameters
        ----------
        iris : pd.DataFrame
            DataFrame with Iris dataset columns possibly renamed for testing.
        iris_cols : pd.Series
            Series derived from the DataFrame to test automatic formatting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col = iris_cols.name
        new_col = f"{col} %"
        df_w_pcnt_sign_col = iris.rename(columns={col: new_col})
        try:
            an_expected_value = f"{df_w_pcnt_sign_col[new_col].iloc[0]:.0%}"
        except ValueError:
            # Can't format str as percent
            # Just make sure nothing fails
            an_expected_value = ""
        styler = display_and_save_df(
            df_w_pcnt_sign_col,
            percentage_format_subset="auto",
            percentage_format_precision=0,
            save_image=save_image,
            save_excel=save_excel,
        )

        html = styler.to_html()
        assert an_expected_value in html

    def test_auto_thousands_format(
        self,
        iris: pd.DataFrame,
        iris_cols: pd.Series,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test automatic thousands separator formatting based on DataFrame column dtype.

        This test checks the functionality that automatically applies thousands
        separators to numeric columns in a DataFrame that are suited for such
        formatting based on column values.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing, which may contain numeric columns.
        iris_cols : pd.Series
            A specific series from the DataFrame to apply the thousands formatting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        iris, iris_cols = iris.copy(), iris_cols.copy()
        col = iris_cols.name

        if pd.api.types.is_numeric_dtype(iris_cols):
            # Make thousands-formatable values
            iris[col] = (iris_cols * 10**6).astype(int)
            an_expected_value = f"{iris[col].iloc[0]:,}"
        else:
            # Can't .astype(int) to str column
            # Just make sure nothing fails
            an_expected_value = ""

        styler = display_and_save_df(
            iris,
            thousands_format_subset="auto",
            save_image=save_image,
            save_excel=save_excel,
        )

        assert an_expected_value in styler.to_html()

    @pytest.mark.parametrize("precision", range(3))
    def test_auto_float_format(
        self,
        iris: pd.DataFrame,
        iris_cols: pd.Series,
        precision: int,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Examine the application of floating-point formatting across various precisions.

        This test assesses the `display_and_save_df` function's capability to format
        floating point numbers within a DataFrame according to specified precision
        levels, ensuring correctness in both the display and potential file outputs.

        Parameters
        ----------
        iris : pd.DataFrame
            DataFrame containing numeric data for precision testing.
        iris_cols : pd.Series
            The specific series used to test floating-point precision.
        precision : int
            The number of decimal places to format the float values.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        iris, iris_cols = iris.copy(), iris_cols.copy()

        if pd.api.types.is_numeric_dtype(iris_cols):
            an_expected_value = f"{iris_cols.iloc[0]:.{precision}f}"
        else:
            # Can't format str as float
            # Just make sure nothing fails
            an_expected_value = ""

        styler = display_and_save_df(
            iris,
            float_format_subset="auto",
            float_format_precision=precision,
            save_image=save_image,
            save_excel=save_excel,
        )

        assert an_expected_value in styler.to_html()

    combos = [
        list(combo) for r in range(4) for combo in itertools.combinations("ABC", r)
    ]

    @pytest.mark.parametrize("subset", combos, ids=map(repr, combos))  # noqa
    def test_datetime_format(
        self,
        datetime_df: pd.DataFrame,
        subset: Sequence[str],
        strftime: str,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Validate date formatting in a DataFrame using a variety of strftime patterns.

        This test ensures that dates within a DataFrame are formatted correctly
        according to a given strftime pattern. It applies this formatting to different
        subsets of columns and checks the resulting HTML output of the styled DataFrame
        for the expected formatted date.

        Parameters
        ----------
        datetime_df : pd.DataFrame
            A DataFrame containing date columns to format.
        subset : Sequence[str]
            The columns of the DataFrame to format.
        strftime : str
            The strftime pattern to apply for formatting dates.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        if len(subset) > 0:
            an_expected_value = datetime_df[subset[0]].iloc[0]
            an_expected_value = f"{an_expected_value:{strftime}}"
        else:
            # If no subset is given, it's hard to say how exactly
            # the dates will be formatted. For now, just make sure
            # that nothing fails.
            an_expected_value = ""

        styler = display_and_save_df(
            datetime_df,
            date_format_subset=subset,
            date_format=strftime,
            save_image=save_image,
            save_excel=save_excel,
        )
        html = styler.to_html()
        assert an_expected_value in html

    def test_auto_datetime_format(
        self,
        datetime_and_float_df: pd.DataFrame,
        strftime: str,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Check automatic date formatting in a DataFrame with varying strftime patterns.

        This test automates the application of date formatting across DataFrame columns,
        evaluating how different strftime patterns are applied automatically and
        verifying the inclusion of formatted dates in the HTML output of the styled
        DataFrame.

        Parameters
        ----------
        datetime_and_float_df : pd.DataFrame
            The DataFrame used for testing, which includes date columns.
        strftime : str
            The strftime pattern to use for formatting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        expected_values = [
            f"{datetime_and_float_df[col].iloc[-1]:{strftime}}" for col in "ABC"
        ]

        styler = display_and_save_df(
            datetime_and_float_df,
            date_format_subset="auto",
            date_format=strftime,
            save_image=save_image,
            save_excel=save_excel,
        )

        html = styler.to_html()
        assert all(expected_value in html for expected_value in expected_values)

    def test_hide_index(
        self, iris: pd.DataFrame, save_image: bool, save_excel: bool
    ) -> None:
        """
        Confirm the ability to effectively hide or display the DataFrame index.

        This test checks whether the index of a DataFrame can be hidden from the
        display and saved outputs, ensuring that both the visibility and omission of the
        index are handled correctly when specified.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing the hide index functionality.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        iris = iris.copy().set_index("Name")
        styler = display_and_save_df(
            iris, hide_index=True, save_image=save_image, save_excel=save_excel
        )
        html = styler.to_html()

        assert all(name not in html for name in iris.index.unique())

        styler_with_index = display_and_save_df(
            iris, hide_index=False, save_image=save_image, save_excel=save_excel
        )
        html_with_index = styler_with_index.to_html()

        assert all(name in html_with_index for name in iris.index.unique())

    def test_ryg_background_gradient(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test red-yellow-green gradient background styling in DataFrame displays.

        This method verifies the correct application of a red-yellow-green (RYG)
        gradient background to DataFrame cells based on their values. It checks whether
        numeric and non-numeric columns appropriately trigger styling effects or raise
        errors.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing gradient background styling.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns to apply the gradient styling to.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        # Should get ValueError on string dtypes
        if pd.api.types.is_string_dtype(iris[col_name]):
            context: ContextManager[object | None] = pytest.raises(
                ValueError, match="could not convert string to float"
            )
        else:
            context = tm.does_not_raise()

        with context:
            styler = display_and_save_df(
                iris,
                ryg_bg_subset=iris_single_col_subset,
                save_image=save_image,
                save_excel=save_excel,
            )
            html = styler.to_html()
            expected_min_color = "background-color: #f8696b"
            expected_max_color = "background-color: #63be7b"
            assert expected_min_color.casefold() in html.casefold()
            assert expected_max_color.casefold() in html.casefold()

    def test_ryg_background_vmin(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Assess the RYG background gradient's response to a set minimum value (`vmin`).

        This test evaluates how setting a `vmin` influences the rendering of the
        red-yellow-green background gradient in a DataFrame. It ensures that values
        below the `vmin` do not affect the gradient's lower color spectrum.

        Parameters
        ----------
        iris : pd.DataFrame
            DataFrame used for gradient styling testing.
        iris_single_col_subset : str | pd.Index
            Specific columns to which the gradient is applied.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        # All iris columns should have min > 0
        assert all(iris.select_dtypes("number").min() > 0)

        # By setting vmin=0, the bottom red color should not appear.
        styler = display_and_save_df(
            iris,
            ryg_bg_subset=iris_single_col_subset,
            ryg_bg_vmin=0,
            save_image=save_image,
            save_excel=save_excel,
        )

        expected_min_color = "background-color: #f8696b"
        expected_max_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() not in html
        assert expected_max_color.casefold() in html

    def test_ryg_background_vmax(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test setting a minimum value for bar background gradient in DataFrame displays.

        This test assesses how setting a lower bound (`vmin`) for bar styles affects the
        visual representation of data within DataFrame cells. It checks that bars are
        proportionally scaled to reflect this setting, ensuring correct visual data
        representation.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing the bar style minimum value impact.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns affected by the vmin setting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return
        # a vmax larger than any value:
        vmax = iris.max(numeric_only=True).max() + 1
        # By settings a large vmax, the top red color should not appear
        styler = display_and_save_df(
            iris,
            ryg_bg_subset=iris_single_col_subset,
            ryg_bg_vmax=vmax,
            save_image=save_image,
            save_excel=save_excel,
        )

        expected_min_color = "background-color: #f8696b"
        expected_max_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() in html
        assert expected_max_color.casefold() not in html

    def test_gyr_background_vmin(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test the green-yellow-red background gradient's response to a set minimum value.

        This method evaluates how setting a minimum threshold affects the rendering of
        the green-yellow-red (GYR) background gradient, ensuring that values below this
        threshold do not influence the gradient's lowest color range.

        Parameters
        ----------
        iris : pd.DataFrame
            DataFrame used for gradient styling testing.
        iris_single_col_subset : str | pd.Index
            The specific columns to which the gradient is applied.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        # All iris columns should have min > 0
        assert all(iris.select_dtypes("number").min() > 0)

        # By setting vmin=0, the bottom green color should not appear.
        styler = display_and_save_df(
            iris,
            gyr_bg_subset=iris_single_col_subset,
            gyr_bg_vmin=0,
            save_image=save_image,
            save_excel=save_excel,
        )

        expected_max_color = "background-color: #f8696b"
        expected_min_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() not in html
        assert expected_max_color.casefold() in html

    def test_gyr_background_vmax(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Validate the effects of setting a maximum value for GYR background styling.

        This test checks how applying an upper limit affects the green-yellow-red
        gradient background in DataFrame cells. It ensures that values above this limit
        are capped at the maximum color range.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame to test with gradient background styling.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns to which the gradient is applied.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return
        # a vmax larger than any value:
        vmax = iris.max(numeric_only=True).max() + 1
        # By settings a large vmax, the top red color should not appear
        styler = display_and_save_df(
            iris,
            gyr_bg_subset=iris_single_col_subset,
            gyr_bg_vmax=vmax,
            save_image=save_image,
            save_excel=save_excel,
        )

        expected_max_color = "background-color: #f8696b"
        expected_min_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() in html
        assert expected_max_color.casefold() not in html

    def test_bar_style(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Validate bar style formatting in DataFrame displays.

        This test ensures that bar styling is correctly applied to DataFrame columns,
        visually representing data magnitude directly in the cells as bar lengths. It
        focuses on numeric columns to check if the bars are displayed correctly and
        accounts for the non-applicability on string type columns.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing bar style formatting.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns to apply the bar styling to.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            # Can't apply bar format to string dtype columns
            return
        styler = display_and_save_df(
            iris,
            bar_subset=iris_single_col_subset,
            save_image=save_image,
            save_excel=save_excel,
        )
        html = styler.to_html()
        expected_max_bar = (
            "background: linear-gradient" "(90deg, #638ec6 90.0%, transparent 90.0%);"
        ).casefold()

        assert expected_max_bar in html.casefold()

    @staticmethod
    def bar_pcnt_from_html(html: str) -> pd.Series:
        """
        Extract bar percentage values from HTML style attributes in a styled DataFrame.

        This utility function parses the HTML output of a styled DataFrame to extract
        numerical percentage values that represent the width of bars used in bar
        formatting. It returns these percentages as a `pandas` Series for further
        analysis or testing.

        Parameters
        ----------
        html : str
            The HTML string from which to extract bar percentages.

        Returns
        -------
        pd.Series
            A Series containing the extracted percentage values as floats.
        """
        regexp = (
            r"background: linear\-gradient\(90deg,.* #638ec6 "
            r"(?P<percentage>\d{1,2}\.\d)\%"
        )
        pcnts = re.findall(regexp, html)
        series = pd.Series(pcnts, dtype=float)
        return series

    def test_bar_vmin(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test the impact of setting a minimum value for bar styles in DataFrame displays.

        This test assesses how setting a lower bound (`vmin`) for bar styles affects the
        visual representation of data within DataFrame cells. It checks that bars are
        proportionally scaled to reflect this setting, ensuring correct visual data
        representation.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing the bar style minimum value impact.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns affected by the vmin setting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        regular_styler = display_and_save_df(
            iris,
            bar_subset=iris_single_col_subset,
            save_image=save_image,
            save_excel=save_excel,
        )
        vmin_styler = display_and_save_df(
            iris,
            bar_subset=iris_single_col_subset,
            bar_vmin=-1,
            save_image=save_image,
            save_excel=save_excel,
        )

        # By setting a vmin smaller than the column min, the bars should get larger
        # Except for the maximum
        regular_pcnts = self.bar_pcnt_from_html(regular_styler.to_html())
        vmin_pcnts = self.bar_pcnt_from_html(vmin_styler.to_html())
        assert (vmin_pcnts >= regular_pcnts).all()

    def test_bar_vmax(
        self,
        iris: pd.DataFrame,
        iris_single_col_subset: str | pd.Index,
        save_image: bool,
        save_excel: bool,
    ) -> None:
        """
        Test setting a maximum value for bar styles in DataFrame displays.

        This method evaluates the changes in bar style visualization when an upper limit
        (`vmax`) is applied to the data values represented in DataFrame bars. It checks
        for the correct scaling of bars to ensure they do not exceed the specified
        maximum.

        Parameters
        ----------
        iris : pd.DataFrame
            The DataFrame used for testing the bar style maximum value impact.
        iris_single_col_subset : str | pd.Index
            The column or subset of columns affected by the vmax setting.
        save_image : bool
            If True, saves the output as an image.
        save_excel : bool
            If True, saves the output as an Excel file. Requires `openpyxl`.
        """
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        regular_styler = display_and_save_df(
            iris,
            bar_subset=iris_single_col_subset,
            save_image=save_image,
            save_excel=save_excel,
        )
        vmax_styler = display_and_save_df(
            iris,
            bar_subset=iris_single_col_subset,
            bar_vmax=10,
            save_image=save_image,
            save_excel=save_excel,
        )

        # By setting a vmax larger than the global max, the bars should get scaled down
        regular_pcnts = self.bar_pcnt_from_html(regular_styler.to_html())
        vmax_pcnts = self.bar_pcnt_from_html(vmax_styler.to_html())
        assert (vmax_pcnts < regular_pcnts).all()

    # TODO: tests of additional *_kwargs. Should somehow parameterize all the tests
    #  above, if possible...


@pytest.mark.flaky(rerun_filter=tm._is_tkinter_error, max_runs=5)
@pytest.mark.parametrize("save_image", [True, False], ids="save_image={}".format)
class TestDisplayAndSaveFig:
    """
    Tests for the `display_and_save_fig` function.

    This suite tests the functionality to display and save `matplotlib` Figure and Axes
    objects as images, ensuring that the outputs are generated without error and match
    expected visual outputs. It includes tests for filename inference from Figure and
    Axes titles and hash-based filename generation for uniqueness.

    Methods
    -------
    display_and_save_fig(*args, **kwargs):
        Wraps the original function to minimize tkagg windows.
    test_doesnt_fail(mpl_figs_and_axes, save_image):
        Ensures the function handles `matplotlib` objects without throwing exceptions.
    test_expected_output(save_image, iris, tmp_path):
        Compares a generated plot against a pre-saved reference image for consistency.
    test_infer_filename_from_fig_suptitle(save_image, mpl_figs_and_axes):
        Checks if the filename is correctly inferred from the Figure's suptitle.
    test_infer_filename_from_axes(save_image, mpl_figs_and_axes):
        Verifies that the filename can be derived from the title of an Axes object.
    test_filename_from_hash(save_image, mpl_figs_and_axes):
        Tests filename generation based on a hash of the Figure or Axes content.
    """

    def display_and_save_fig(self, *args: Any, **kwargs: Any) -> None:
        """
        Create a wrapper around the original `display_and_save_fig` function.

        This method handles the saving of `matplotlib` figures or axes as image files,
        wrapping the original function with additional logic to minimize the graphical
        backend overhead, particularly for automated testing environments.

        Parameters
        ----------
        *args : Any
            Arguments passed to the original `display_and_save_fig`.
        **kwargs : Any
            Keyword arguments passed to the original `display_and_save_fig`.
        """
        from EAB_tools.io.display import display_and_save_fig as display_and_save_fig_og

        display_and_save_fig_og(*args, **kwargs)
        tm._minimize_tkagg()

    def test_doesnt_fail(
        self, mpl_figs_and_axes: plt.Figure | plt.Axes, save_image: bool
    ) -> None:
        """
        Ensure that the `display_and_save_fig` function executes without errors.

        This test verifies that the function can handle `matplotlib` Figure or Axes
        objects without raising exceptions, including the option to save the output as
        an image.

        Parameters
        ----------
        mpl_figs_and_axes : plt.Figure | plt.Axes
            The `matplotlib` figure or axes to test.
        save_image : bool
            Indicates whether the output should be saved as an image.
        """
        display_and_save_fig(mpl_figs_and_axes, save_image=save_image)

    def test_expected_output(
        self,
        save_image: bool,
        iris: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """
        Validate that the output of `display_and_save_fig` matches the expected result.

        This method tests the function with a specific `matplotlib` plot created from
        the Iris dataset, comparing the saved image against a reference image to ensure
        consistency.

        Parameters
        ----------
        save_image : bool
            If true, the plot is saved to an image file.
        iris : pd.DataFrame
            The Iris dataset used to generate the plot.
        tmp_path : Path
            The temporary directory path where the image is saved for comparison.
        """
        fig, ax = plt.subplots(
            subplot_kw={
                "xlabel": "SepalLength",
                "ylabel": "SepalWidth",
                "title": self.test_expected_output.__name__,
            },
            facecolor="white",
        )

        for iris_type in iris["Name"].unique():
            subset: pd.DataFrame = iris[iris["Name"] == iris_type]
            ax.plot("SepalLength", "SepalWidth", "o", data=subset, label=iris_type)
        ax.legend()

        display_and_save_fig(fig, save_image=True, filename="foo.png")
        assert tm._test_photos_are_equal(
            tmp_path / "foo.png",
            Path(__file__).parent / "data" / "test_expected_output.png",
        )
        plt.close(fig)

    def test_infer_filename_from_fig_suptitle(
        self,
        save_image: bool,
        mpl_figs_and_axes: plt.Figure | plt.Axes,
    ) -> None:
        """
        Test automatic filename generation from the figure's suptitle when saving.

        This method checks if `display_and_save_fig` correctly infers the filename from
        the figure's suptitle and saves the image file accordingly.

        Parameters
        ----------
        save_image : bool
            Determines whether to save the figure as an image.
        mpl_figs_and_axes : plt.Figure | plt.Axes
            The `matplotlib` figure or axes from which the filename is inferred.
        """
        fig: plt.Figure
        if isinstance(mpl_figs_and_axes, plt.Axes):
            get_figure = mpl_figs_and_axes.get_figure()
            assert get_figure is not None  # for `mypy`
            fig = get_figure
        else:
            fig = mpl_figs_and_axes

        name = sanitize_filename(str(fig))
        fig.suptitle(name)

        display_and_save_fig(mpl_figs_and_axes, save_image=True)
        assert Path(f"{name}.png").exists()

    def test_infer_filename_from_axes(
        self, save_image: bool, mpl_figs_and_axes: plt.Figure | plt.Axes
    ) -> None:
        """
        Verify filename determination from `Axes` title during figure saving.

        This test evaluates the functionality of `display_and_save_fig` to derive the
        filename from the title of an `Axes` object in the figure or axes object and
        saves it as an image file if requested.

        Parameters
        ----------
        save_image : bool
            If True, the image is saved under the derived filename.
        mpl_figs_and_axes : plt.Figure | plt.Axes
            The `matplotlib` object containing the axis with the title.
        """
        if isinstance(mpl_figs_and_axes, plt.Figure):
            axes: plt.Axes = mpl_figs_and_axes.axes[0]
        else:
            axes = mpl_figs_and_axes

        # Pretty random name
        name = sanitize_filename(str(mpl_figs_and_axes))
        axes.set_title(name)

        display_and_save_fig(mpl_figs_and_axes, save_image=True)
        assert Path(f"{name}.png").exists()

    def test_filename_from_hash(
        self,
        save_image: bool,
        mpl_figs_and_axes: plt.Figure | plt.Axes,
    ) -> None:
        """
        Assess the hash-based filename generation for saved figures.

        This test checks whether `display_and_save_fig` can successfully generate a
        filename based on the hash of the `matplotlib` `Figure` or `Axes` content, and
        then save the image accordingly.

        Parameters
        ----------
        save_image : bool
            Indicates if the figure should be saved as an image.
        mpl_figs_and_axes : plt.Figure | plt.Axes
            The `matplotlib` figure or axes to hash for filename generation.
        """
        expected_hash = eab.util.hash_mpl_fig(mpl_figs_and_axes)
        display_and_save_fig(mpl_figs_and_axes, save_image=True)

        assert Path(f"{expected_hash}.png").exists()
