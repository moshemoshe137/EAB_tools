# Assortment of functions, stolen from Fa21 data review and tweaked for this project

import dataframe_image as dfi
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.formats.style
from pandas import Series, DataFrame, Index, MultiIndex

import hashlib
from pathlib import Path
import re


def sanitize_filename(filename):
    return re.sub(r'[^\w\-_. ()]', '_', filename)


def hash_df(df, styler=None, max_len=5, date=False):
    # Ensure they passed the correct datatype
    if not isinstance(df, (DataFrame, Index, MultiIndex, Series)):
        # In particular, suggest using hash_mpl_fig() for
        # matplotlib objects
        if isinstance(df, (plt.Axes, plt.Figure)):
            raise TypeError(f"Use hash_mpl_fig() instead.")
        # For all other dtypes, raise TypeError
        raise TypeError(f"Type of {type(df)} is not understood.")

    # Hash the dataframe
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df).values)
    h.update(pd.util.hash_pandas_object(df.columns).values)

    # Optionally, hash the Styler as well so that
    # DataFrames with the same data but different styles
    # will have different hashes.
    if styler is not None:
        h.update(
            # Both `hash(styler)` and `hash(styler.to_html())`
            # produced inconsistent hashes.
            styler.to_latex(convert_css=True).encode('utf-8')
        )

    # Optionally, concat the date and time to the beginning of the hash.
    # This will completely prevent overwriting and will produce a new set
    # of output objects every time.
    date = pd.Timestamp.now().strftime('%#m.%#d.%y %H%M%S ') if date else ''
    return date + h.hexdigest()[:max_len]


def display_and_save_df(
        df, caption=None, filename=None,
        large_title=True, large_header=True, cell_borders=True,
        highlight_total_row=False, border_width='1px',
        thousands_format_subset='auto',
        date_format_subset='auto', date_format="%#m/%#d/%Y",
        percentage_format_subset='auto', percentage_format_precision=1,
        float_format_subset='auto', float_format_precision=1,
        hide_index=False, convert_dtypes=True,
        ryg_bg_subset=None, ryg_bg_vmin=None, ryg_bg_vmax=None,
        gyr_bg_subset=None, gyr_bg_vmin=None, gyr_bg_vmax=None,
        bar_subset=None, bar_vmin=None, bar_vmax=None,
        format_kwargs=[], background_gradient_kwargs=[], bar_kwargs=[],
        save_excel=False, save_image=True,
        min_width='10em', max_width="25em"
):
    # Determine the DataFrame and Styler objects
    # This function accepts an instance of DataFrame, Series, or Styler.
    if isinstance(df, Series):
        df = df.to_frame()

    if isinstance(df, pd.io.formats.style.Styler):
        df, styler = df.data, df
    else:
        styler = df.style

    if convert_dtypes:
        df = df.convert_dtypes()

    # Define some styles. These do not export to Excel.
    LARGE_TITLE = [{
        'selector': 'caption',
        'props': [('font-size', '225%'), ('text-align', 'center')]
    }]
    LARGE_HEADER = [{
        'selector': 'th',
        'props': [('font-size', '110%'), ('border-style', 'solid'),
                  ('text-align', 'center')]
    }]
    CELL_BORDERS = [{
        'selector': 'td',  # 'td:not(tr:last-child)',
        'props': [('border-style', 'solid'), ('border-width', border_width)]
    }, {
        'selector': 'th',
        'props': [('border-style', 'solid'), ('border-width', border_width)]
    }]

    HIGHLIGHT_TOTAL = [{
        'selector': 'tr:last-child',
        'props': [('font-weight', 'bold'), ('font-size', '110%')]
    }]

    MIN_WIDTH = [{
        'selector': 'td,th',
        'props': [('min-width', min_width), ('max-width', max_width)]
        # Defaults to "10em" based on hardcoded value in 
        # Styler.bar(). https://github.com/pandas-dev/pandas/blob/v1.3.2/pandas/io/formats/style.py#L2144
    }]
    styler = styler.set_table_styles(MIN_WIDTH)

    # Add caption
    styler.set_caption(caption)

    # Apply styles
    if large_title:
        styler = styler.set_table_styles(LARGE_TITLE, overwrite=False)
    if large_header:
        styler = styler.set_table_styles(LARGE_HEADER, overwrite=False)
    if cell_borders:
        styler = styler.set_table_styles(CELL_BORDERS, overwrite=False)
    if highlight_total_row:
        styler = styler.set_table_styles(HIGHLIGHT_TOTAL, overwrite=False)

    # Apply formats
    if percentage_format_subset is not None:
        if isinstance(percentage_format_subset, str) and \
                percentage_format_subset == 'auto':
            try:
                percentage_format_subset = df.columns[df.columns.str.contains('%')]
            except AttributeError as e:
                # Can only use .str accessor with Index, not MultiIndex
                percentage_format_subset = []
                print(e)
        formatter = f'{{:.{percentage_format_precision}%}}'
        styler = styler.format(formatter=formatter,
                               subset=percentage_format_subset)

    if thousands_format_subset is not None:
        if thousands_format_subset == 'auto':
            thousands_format_subset = df.select_dtypes(int).columns
        styler = styler.format("{:,.0f}",
                               subset=thousands_format_subset,
                               precision=0)

    if float_format_subset is not None:
        if float_format_subset == 'auto':
            float_format_subset = df.select_dtypes(float) \
                .columns \
                .drop(percentage_format_subset, errors='ignore')
        styler = styler.format(
            precision=float_format_precision, subset=float_format_subset)

    if date_format_subset is not None:
        if date_format_subset == 'auto':
            date_format_subset = df.select_dtypes('datetime').columns
        styler = styler.format(f"{{:{date_format}}}",
                               subset=date_format_subset)

    # Hide axes
    if hide_index:
        styler = styler.hide_index()

    # Add simple background gradients
    if ryg_bg_subset is not None:
        styler = styler.background_gradient(
            xl_RYG_cmap, subset=ryg_bg_subset, text_color_threshold=0,
            vmin=ryg_bg_vmin, vmax=ryg_bg_vmax)
    if gyr_bg_subset is not None:
        styler = styler.background_gradient(
            xl_GYR_cmap, subset=gyr_bg_subset, text_color_threshold=0,
            vmin=gyr_bg_vmin, vmax=gyr_bg_vmax)

    if bar_subset is not None:
        styler = styler.bar(subset=bar_subset, color='#638ec6',
                            vmin=bar_vmin, vmax=bar_vmax, width=90)

    # Accepts kwargs as list of dicts to unpack
    for format_kwarg in format_kwargs:
        styler = styler.format(**format_kwarg)
    for background_gradient_kwarg in background_gradient_kwargs:
        styler = styler.background_gradient(text_color_threshold=0, **background_gradient_kwarg)
    for bar_kwarg in bar_kwargs:
        if 'color' not in bar_kwarg:
            # Default bar color, from Excel
            bar_kwarg['color'] = '#638ec6'
        styler = styler.bar(**bar_kwarg)

    if filename is None and styler.caption is not None:
        # If no filename is given, use the caption
        filename = f"DataFrame {styler.caption}.png"
        # Remove bad filename characters from the caption
        # with a little re magic
        # https://stackoverflow.com/a/13593932
        filename = sanitize_filename(filename)

    filename = Path(filename) if filename else Path(f"DataFrame {hash_df(df, styler)}.png")
    Path('output').mkdir(exist_ok=True)
    filepath = 'output' / filename

    excel_output = Path('output/output.xlsx')

    # Determine ExcelWriter params based on if the file exists or not
    mode, ise = ('a', 'replace') if excel_output.exists() else ('w', None)

    # Determine Excel sheet name (sn)
    # Sheet names are capped at 31 chars
    sn = styler.caption[-31:] if styler.caption else str(filename)[-31:]
    sn = sanitize_filename(sn)

    # Now that everything is configured, we can get ready to save the
    # styled DataFrame to the disk. We first display it inline in the
    # Jupyter Notebook using `IPython.display.display()`.
    # Next, Styler is outputted to Excel. First using `Styler.to_excel()`,
    # but then doing quite a bit of additional nitpicking for
    # number formats and conditional formats to output properly.
    # 
    # Finally, the Styler is exported as a png using the
    # `Styler.export_png()` function provided by module
    # dataframe_image.

    # Display with IPython.display.display()
    display(styler)

    # Export to Excel sheet    
    def to_excel():
        with pd.ExcelWriter(excel_output, engine='openpyxl',
                            mode=mode, if_sheet_exists=ise) as wb:
            print(f"Exporting to Excel as '{excel_output.parent}\\"
                  f"[{excel_output.name}]{sn}'", end=f" ... ")
            styler.to_excel(wb, sheet_name=sn, engine='openpyxl')

            if (percentage_format_subset is not None) or (thousands_format_subset is not None) \
                    or (bar_subset is not None):
                # Number formatting doesn't seem to carry over to
                # Excel automatically with pandas. Since percentages, thousands, etc. are 
                # so widespread, I am using openpyxl to convert the number formats.
                # 
                # Additionally, I am using openpyxl to add data bar conditional formatting
                # for bar_subset.

                len_index = df.index.nlevels

                # Get the worksheet
                ws = wb.book[sn]

                # Determine the 0-based pandas indicies of formatted subsets
                pcnt_cols = [df.columns.get_loc(col)
                             for col in percentage_format_subset] if percentage_format_subset is not None else []
                tsnd_cols = [df.columns.get_loc(col)
                             for col in thousands_format_subset] if thousands_format_subset is not None else []
                try:
                    bar_cols = [df.columns.get_loc(col)
                                for col in bar_subset] if bar_subset is not None else []
                except TypeError:
                    # bar_subset contains more than just columns
                    # for now, we must abort trying to export data 
                    # bar conditional formatting.
                    bar_cols = []

                # Iterate through the columns, applying styles to
                # all cells in a column when we arrive at certain columns.
                for col_num, col in enumerate(ws.iter_cols()):
                    if np.any(col_num - len_index in pcnt_cols):
                        for cell in col:
                            code = f"0{'.' * (percentage_format_precision > 0)}{'0' * percentage_format_precision}%"
                            cell.number_format = code
                    if col_num - len_index in tsnd_cols:
                        for cell in col:
                            cell.number_format = f"#,##0"

                # Add bar conditional formatting, using the style's vmin
                # and vmax, if given
                rule = openpyxl.formatting.rule.DataBarRule(
                    start_type='num' if bar_vmin is not None else 'min',
                    start_value=bar_vmin if bar_vmin is not None else None,
                    end_type='num' if bar_vmax is not None else 'max',
                    end_value=bar_vmax if bar_vmax is not None else None,
                    color="638EC6", showValue=True, minLength=None, maxLength=None
                )

                for col in bar_cols:
                    letter = openpyxl.utils.cell.get_column_letter(col + len_index + 1)
                    end_num = len(df) + df.columns.nlevels

                    # I could not even find the function ws.conditional_formatting.add()
                    # in the openpyxl docs. Thank god for https://stackoverflow.com/a/32454012.
                    ws.conditional_formatting.add(f"{letter}1:{letter}{end_num}", rule)

    if save_excel:
        try:
            to_excel()
        except Exception as e:
            print("Error exporting to Excel!")
            print(e)

    if save_image:
        print(f"Saving as '{filepath}'", end=f" ... ")
        styler.export_png = styler.export_png
        styler.export_png(str(filepath), fontsize=16, max_rows=200)
