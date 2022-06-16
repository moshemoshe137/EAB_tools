import os
import warnings
from typing import (
    Any,
    Union,
    Optional,
    Sequence,
)

import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler, Subset


# import openpyxl

# Copied from Excel's conditional formatting Red-Yellow-Green built-in colormap.
_xl_RYG_colors = ["#F8696B", "#FFEB84", "#63BE7B"]
xl_RYG_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("xl_RYG_cmap", _xl_RYG_colors)
xl_GYR_cmap = xl_RYG_cmap.reversed()


def display_and_save_df(
        df: Union[pd.DataFrame, pd.Series, Styler],
        caption: str = '',
        filename: Optional[os.PathLike] = None,
        large_title: bool = True,
        large_col_names: bool = True,
        cell_borders: bool = True,
        highlight_total_row: bool = False,
        border_width: str = '1px',
        thousands_format_subset: Union[Subset, str] = 'auto',
        date_format_subset: Union[Subset, str] = 'auto',
        date_format: str = "%#m/%#d/%Y",
        percentage_format_subset: Union[Subset, str] = 'auto',
        percentage_format_precision: int = 1,
        float_format_subset: Union[Subset, str] = 'auto',
        float_format_precision: int = 1,
        hide_index: bool = False,
        convert_dtypes: bool = True,
        ryg_bg_subset: Optional[Union[Subset, str]] = None,
        ryg_bg_vmin: Optional[int] = None,
        ryg_bg_vmax: Optional[int] = None,
        gyr_bg_subset: Optional[Union[Subset, str]] = None,
        gyr_bg_vmin: Optional[int] = None,
        gyr_bg_vmax: Optional[int] = None,
        bar_subset: Optional[Union[Subset, str]] = None,
        bar_vmin: Optional[int] = None,
        bar_vmax: Optional[int] = None,
        format_kwargs: Sequence[dict[str, Any]] = [],
        background_gradient_kwargs: Sequence[dict[str, Any]] = [],
        bar_kwargs: Sequence[dict[str, Any]] = [],
        save_excel: bool = False,
        save_image: bool = True,
        min_width: str = "10em",
        max_width: str = "25em"
) -> Styler:
    # Convert pd.Series to Frame if needed
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Determine the DataFrame and the Styler
    if isinstance(df, Styler):
        df, styler = df.data, df
    else:
        styler = df.style

    # Convert dtypes if requested
    if convert_dtypes:
        df = df.convert_dtypes()

    # Define some styles.
    # These do NOT export to Excel!
    LARGE_TITLE = dict(
        selector='caption',
        props=[('font-size', '225%'), ('text-align', 'center')])

    LARGE_COL_NAMES = dict(
        selector='th',
        props=[('font-size', '110%'), ('border-style', 'solid'), ('text-align', 'center')])

    CELL_BORDERS = dict(
        selector='th,td',
        props=[('border-style', 'solid'), ('border-width', border_width)])

    HIGHLIGHT_TOTAL = dict(
        selector='tr:last-child',
        props=[('font-weight', 'bold'), ('font-size', '110%')])

    MIN_MAX_WIDTH = dict(
        selector='th',
        props=[('min-width', min_width), ('max-width', max_width)])
    # Mind width defaults to "10em" based on hardcoded value in
    # Styler.bar(). https://github.com/pandas-dev/pandas/blob/9222cb0c/pandas/io/formats/style.py#L3097

    # Enforce min and max width
    styler = styler.set_table_styles([MIN_MAX_WIDTH], overwrite=False)

    # Apply the caption if it is not None
    styler.set_caption(caption)

    # Apply optional styles
    if large_title:
        styler = styler.set_table_styles([LARGE_TITLE], overwrite=False)
    if large_col_names:
        styler = styler.set_table_styles([LARGE_COL_NAMES], overwrite=False)
    if cell_borders:
        styler = styler.set_table_styles([CELL_BORDERS], overwrite=False)
    if highlight_total_row:
        styler = styler.set_table_styles([HIGHLIGHT_TOTAL], overwrite=False)

    # Find auto percentage format columns
    if isinstance(percentage_format_subset, str) and percentage_format_subset == 'auto':
        try:
            flattened_cols = df.columns.to_flat_index().astype('string')
            # Grab the cols that contain percent signs
            percentage_format_subset_mask = flattened_cols.str.contains('%')
            # But don't accept cols that are a string dtype
            percentage_format_subset_mask &= [not pd.api.types.is_string_dtype(df[col])
                                              for col in df]
            percentage_format_subset = df.columns[percentage_format_subset_mask]
        except AttributeError as e:
            # Can only use .str accessor with Index, not MultiIndex
            warnings.warn(e)
            percentage_format_subset = []
    # Apply the percentage format
    if percentage_format_subset is not None:
        formatter = f'{{:.{percentage_format_precision}%}}'
        styler = styler.format(formatter=formatter, subset=percentage_format_subset)

    # Apply thousands seperator
    if thousands_format_subset is not None:
        if thousands_format_subset == 'auto':
            thousands_format_subset = df.select_dtypes(int).columns
        styler = styler.format("{:,}", subset=thousands_format_subset)

    # Apply floating point precision
    if float_format_subset is not None:
        if float_format_subset == 'auto':
            float_format_subset = (df.select_dtypes(float)
                                   .columns
                                   .drop(percentage_format_subset, errors='ignore'))
        styler = styler.format(f"{{:.{float_format_precision}f}}", subset=float_format_subset)

    # Apply date formatting
    if date_format_subset is not None:
        if date_format_subset == 'auto':
            date_format_subset = df.select_dtypes('datetime').columns
        styler = styler.format(f"{{:{date_format}}}", subset=date_format_subset)

    # Hide axes
    if hide_index:
        styler = styler.hide(axis='index')

    # Apply RYG or GYR conditional formatting
    if ryg_bg_subset is not None:
        styler = styler.background_gradient(
            xl_RYG_cmap, subset=ryg_bg_subset, text_color_threshold=0,
            vmin=ryg_bg_vmin, vmax=ryg_bg_vmax)
    if gyr_bg_subset is not None:
        styler = styler.background_gradient(
            xl_GYR_cmap, subset=gyr_bg_subset, text_color_threshold=0,
            vmin=gyr_bg_vmin, vmax=gyr_bg_vmax)

    # Apply the histogram bar conditional formatting
    if bar_subset is not None:
        styler = styler.bar(subset=bar_subset, color="#638ec6",
                            vmin=bar_vmin, vmax=bar_vmax, width=90)

    return styler
