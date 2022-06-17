import itertools
import os
import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import (
    Union,
    Any,
    Sequence
)

import pandas as pd
import pytest

from EAB_tools.io.io import (
    display_and_save_df
)


@pytest.mark.parametrize('save_image',
                         [
                             pytest.param(True, marks=pytest.mark.slow),
                             False
                         ], ids="save_image={0}".format)
@pytest.mark.parametrize('save_excel', [True, False], ids="save_excel={0}".format)
class TestDisplayAndSave:
    @pytest.fixture(autouse=True)
    def _init(self, tmp_path: Path):
        os.chdir(tmp_path)

    def test_doesnt_fail(self, iris: pd.DataFrame, save_image, save_excel):
        display_and_save_df(iris, save_image=save_image, save_excel=save_excel)

    def test_series_doesnt_fail(self, series: pd.Series, save_image, save_excel):
        display_and_save_df(series, save_image=save_image, save_excel=save_excel)

    def test_multiindex_index(self, iris: pd.DataFrame, save_image, save_excel):
        iris_mi = iris.set_index('Name', append=True)
        display_and_save_df(iris_mi, save_image=save_image, save_excel=save_excel)
        display_and_save_df(iris_mi.T, save_image=save_image, save_excel=save_excel)

    @staticmethod
    def col_name_from_iris_single_col_subset(col_name: Union[str, pd.Index]):
        return col_name if isinstance(col_name, str) else col_name[0]

    @pytest.mark.parametrize(
        "kwargs", [
            pytest.param(dict(format="{:,}", display_and_save_kw="thousands_format_subset",
                              need_large_numbers=True), id='thousands subset'),
            pytest.param(dict(format="{:.1%}", display_and_save_kw="percentage_format_subset",
                              percentage_format_precision=1), id='percentage subset'),
            pytest.param(dict(format="{:.0f}", display_and_save_kw="float_format_subset",
                              float_format_precision=0), id='float.0f'),
            pytest.param(dict(format="{:.1f}", display_and_save_kw="float_format_subset",
                              float_format_precision=1), id='float.1f'),
            pytest.param(dict(format="{:.2f}", display_and_save_kw="float_format_subset",
                              float_format_precision=2), id='float.2f')
        ]
    )
    def test_styler_expected_text(
            self,
            iris: pd.DataFrame,
            iris_single_col_subset: Union[str, pd.Index],
            kwargs: dict[str, Any], save_image, save_excel
    ):
        """Test for expected text in Stylers"""
        iris, kwargs = iris.copy(), kwargs.copy()
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if kwargs.pop('need_large_numbers', False):
            # Need large numbers to test thousands seperator
            large_iris = iris.copy()
            numeric_cols = large_iris.select_dtypes('number').columns
            large_iris[numeric_cols] = (large_iris[numeric_cols] * 10 ** 6).astype(int)
            iris = large_iris

        col = iris[col_name]
        an_expected_value = col.iloc[0]
        try:
            an_expected_value = kwargs.pop('format').format(an_expected_value)
        except ValueError:
            # Number formatting can't be applied to str
            # Just make sure everything goes ok
            an_expected_value = ''

        if pd.api.types.is_string_dtype(col):
            context = pytest.raises(ValueError)
        else:
            context = does_not_raise()
        with context:
            # str columns with incorrect format code should
            # throw a ValueError
            d = {kwargs.pop("display_and_save_kw"): iris_single_col_subset}
            styler = display_and_save_df(iris, save_image=save_image, save_excel=save_excel, **d, **kwargs)
            styler.to_html()
            html = styler.to_html()
            assert an_expected_value in html

    def test_auto_percentage_format(self, iris: pd.DataFrame, iris_cols: pd.Series, save_image, save_excel):
        col = iris_cols.name
        new_col = f"{col} %"
        df_w_pcnt_sign_col = iris.rename(columns={col: new_col})
        try:
            an_expected_value = f"{df_w_pcnt_sign_col[new_col].iloc[0]:.0%}"
        except ValueError:
            # Can't format str as percent
            # Just make sure nothing fails
            an_expected_value = ''
        styler = display_and_save_df(df_w_pcnt_sign_col,
                                     percentage_format_subset='auto',
                                     percentage_format_precision=0,
                                     save_image=save_image, save_excel=save_excel)

        html = styler.to_html()
        assert an_expected_value in html

    def test_auto_thousands_format(self, iris: pd.DataFrame, iris_cols: pd.Series, save_image, save_excel):
        iris, iris_cols = iris.copy(), iris_cols.copy()
        col = iris_cols.name

        if pd.api.types.is_numeric_dtype(iris_cols):
            # Make thousands-formattable values
            iris[col] = (iris_cols * 10 ** 6).astype(int)
            an_expected_value = f"{iris[col].iloc[0]:,}"
        else:
            # Can't .astype(int) to str column
            # Just make sure nothing fails
            an_expected_value = ''

        styler = display_and_save_df(iris,
                                     thousands_format_subset='auto',
                                     save_image=save_image, save_excel=save_excel)

        assert an_expected_value in styler.to_html()

    @pytest.mark.parametrize(
        "precision", range(3)
    )
    def test_auto_float_format(self, iris: pd.DataFrame, iris_cols: pd.Series,
                               precision: int, save_image, save_excel):
        iris, iris_cols = iris.copy(), iris_cols.copy()

        if pd.api.types.is_numeric_dtype(iris_cols):
            an_expected_value = f"{iris_cols.iloc[0]:.{precision}f}"
        else:
            # Can't format str as float
            # Just make sure nothing fails
            an_expected_value = ''

        styler = display_and_save_df(iris,
                                     float_format_subset='auto',
                                     float_format_precision=precision,
                                     save_image=save_image, save_excel=save_excel)

        assert an_expected_value in styler.to_html()

    combos = [list(combo)
              for r in range(4)
              for combo in itertools.combinations('ABC', r)]
    @pytest.mark.parametrize("subset", combos, ids=map(repr, combos))
    def test_datetime_format(self,
                             datetime_df: pd.DataFrame,
                             subset: Sequence[str],
                             strftime: str, save_image, save_excel):
        if len(subset) > 0:
            an_expected_value = datetime_df[subset[0]].iloc[0]
            an_expected_value = f"{an_expected_value:{strftime}}"
        else:
            # If no subset is given, it's hard to say how exactly
            # the dates will be formatted. For now, just make sure
            # that nothing fails.
            an_expected_value = ''

        styler = display_and_save_df(datetime_df,
                                     date_format_subset=subset,
                                     date_format=strftime,
                                     save_image=save_image, save_excel=save_excel)
        html = styler.to_html()
        assert an_expected_value in html

    def test_auto_datetime_format(self,
                                  datetime_and_float_df: pd.DataFrame,
                                  strftime: str, save_image, save_excel):
        expected_values = [f"{datetime_and_float_df[col].iloc[-1]:{strftime}}"
                           for col in 'ABC']

        styler = display_and_save_df(datetime_and_float_df,
                                     date_format_subset='auto',
                                     date_format=strftime,
                                     save_image=save_image, save_excel=save_excel)

        html = styler.to_html()
        assert all(expected_value in html
                   for expected_value in expected_values)

    def test_hide_index(self, iris: pd.DataFrame, save_image, save_excel):
        iris = iris.copy().set_index('Name')
        styler = display_and_save_df(iris, hide_index=True, save_image=save_image, save_excel=save_excel)
        html = styler.to_html()

        assert all(name not in html
                   for name in iris.index.unique())

        styler_with_index = display_and_save_df(iris, hide_index=False, save_image=save_image, save_excel=save_excel)
        html_with_index = styler_with_index.to_html()

        assert all(name in html_with_index
                   for name in iris.index.unique())

    def test_ryg_background_gradient(self,
                                     iris: pd.DataFrame,
                                     iris_single_col_subset: Union[str, pd.Index],
                                     save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        # Should get ValueError on string dtypes
        context = pytest.raises(ValueError, match="could not convert string to float") \
            if pd.api.types.is_string_dtype(iris[col_name]) \
            else does_not_raise()

        with context:
            styler = display_and_save_df(iris,
                                         ryg_bg_subset=iris_single_col_subset,
                                         save_image=save_image, save_excel=save_excel)
            html = styler.to_html()
            expected_min_color = "background-color: #f8696b"
            expected_max_color = "background-color: #63be7b"
            assert expected_min_color.casefold() in html.casefold()
            assert expected_max_color.casefold() in html.casefold()

    def test_ryg_background_vmin(self,
                                 iris: pd.DataFrame,
                                 iris_single_col_subset: Union[str, pd.Index],
                                 save_image, save_excel):

        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        # All iris columns should have min > 0
        assert all(iris.select_dtypes('number').min() > 0)

        # By setting vmin=0, the bottom red color should not appear.
        styler = display_and_save_df(iris, ryg_bg_subset=iris_single_col_subset,
                                     ryg_bg_vmin=0, save_image=save_image, save_excel=save_excel)

        expected_min_color = "background-color: #f8696b"
        expected_max_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() not in html
        assert expected_max_color.casefold() in html

    def test_ryg_background_vmax(self,
                                 iris: pd.DataFrame,
                                 iris_single_col_subset: Union[str, pd.Index],
                                 save_image, save_excel):

        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return
        # a vmax larger than any value:
        vmax = iris.max(numeric_only=True).max() + 1
        # By settings a large vmax, the top red color should not appear
        styler = display_and_save_df(iris, ryg_bg_subset=iris_single_col_subset,
                                     ryg_bg_vmax=vmax, save_image=save_image, save_excel=save_excel)

        expected_min_color = "background-color: #f8696b"
        expected_max_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() in html
        assert expected_max_color.casefold() not in html

    def test_gyr_background_vmin(self,
                                 iris: pd.DataFrame,
                                 iris_single_col_subset: Union[str, pd.Index],
                                 save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        # All iris columns should have min > 0
        assert all(iris.select_dtypes('number').min() > 0)

        # By setting vmin=0, the bottom green color should not appear.
        styler = display_and_save_df(iris, gyr_bg_subset=iris_single_col_subset,
                                     gyr_bg_vmin=0, save_image=save_image, save_excel=save_excel)

        expected_max_color = "background-color: #f8696b"
        expected_min_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() not in html
        assert expected_max_color.casefold() in html

    def test_gyr_background_vmax(self,
                                 iris: pd.DataFrame,
                                 iris_single_col_subset: Union[str, pd.Index],
                                 save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return
        # a vmax larger than any value:
        vmax = iris.max(numeric_only=True).max() + 1
        # By settings a large vmax, the top red color should not appear
        styler = display_and_save_df(iris, gyr_bg_subset=iris_single_col_subset,
                                     gyr_bg_vmax=vmax, save_image=save_image, save_excel=save_excel)

        expected_max_color = "background-color: #f8696b"
        expected_min_color = "background-color: #63be7b"
        html = styler.to_html()

        assert expected_min_color.casefold() in html
        assert expected_max_color.casefold() not in html

    def test_bar_style(self, iris: pd.DataFrame, iris_single_col_subset: Union[str, pd.Index], save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            # Can't apply bar format to string dtype columns
            return
        styler = display_and_save_df(iris, bar_subset=iris_single_col_subset,
                                     save_image=save_image, save_excel=save_excel)
        html = styler.to_html()
        expected_max_bar = "background: linear-gradient(90deg, #638ec6 90.0%, transparent 90.0%);".casefold()

        assert expected_max_bar in html.casefold()

    @staticmethod
    def bar_pcnt_from_html(html: str):
        regexp = r"background: linear\-gradient\(90deg,.* #638ec6 (?P<percentage>\d{1,2}\.\d)\%"
        pcnts = re.findall(regexp, html)
        series = pd.Series(pcnts, dtype=float)
        return series

    def test_bar_vmin(self, iris: pd.DataFrame, iris_single_col_subset: Union[str, pd.Index], save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        regular_styler = display_and_save_df(iris, bar_subset=iris_single_col_subset,
                                             save_image=save_image, save_excel=save_excel)
        vmin_styler = display_and_save_df(iris, bar_subset=iris_single_col_subset, bar_vmin=-1,
                                          save_image=save_image, save_excel=save_excel)

        # By setting a vmin smaller than the column min, the bars should get larger
        # Except for the maximum
        regular_pcnts = self.bar_pcnt_from_html(regular_styler.to_html())
        vmin_pcnts = self.bar_pcnt_from_html(vmin_styler.to_html())
        assert (vmin_pcnts >= regular_pcnts).all()

    def test_bar_vmax(self, iris: pd.DataFrame, iris_single_col_subset: Union[str, pd.Index], save_image, save_excel):
        col_name = self.col_name_from_iris_single_col_subset(iris_single_col_subset)

        if pd.api.types.is_string_dtype(iris[col_name]):
            return

        regular_styler = display_and_save_df(iris, bar_subset=iris_single_col_subset,
                                             save_image=save_image, save_excel=save_excel)
        vmax_styler = display_and_save_df(iris, bar_subset=iris_single_col_subset, bar_vmax=10,
                                          save_image=save_image, save_excel=save_excel)

        # By setting a vmax larger than the global max, the bars should get shorter
        regular_pcnts = self.bar_pcnt_from_html(regular_styler.to_html())
        vmax_pcnts = self.bar_pcnt_from_html(vmax_styler.to_html())
        assert (vmax_pcnts < regular_pcnts).all()

    # TODO: tests of additional *_kwargs. Should somehow parameterize all the tests above, if possible...
