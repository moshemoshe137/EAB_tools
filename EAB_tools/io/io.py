"""Methods to load data from disk."""

from pathlib import Path
from typing import (
    Any,
    Optional,
)

import pandas as pd

from .display import PathLike


def load_df(
    filepath: PathLike,
    file_type: str = "detect",
    cache: bool = True,
    pkl_name: Optional[PathLike] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV or Excel file as a pandas DataFrame.

    Load a datafile ('csv', 'xls', or 'xlsx' file) as a pandas DataFrame. By default,
    cache the datafile for quicker load time next time.

    Parameters
    ----------
    filepath : PathLike
        The path to the datafile
    file_type : {'CSV', 'XLS', 'XLSX', 'detect'}, default 'detect'
        If 'detect', attempt to detect the correct pandas function to read the file
        based on its suffixes. Otherwise, the file type may be manually specified as
        one of {'CSV', 'XLS', 'XLSX'}. The parameter is case-insensitive and may begin
        with a dot.
    cache : bool, default True
        If True and the pickle file exists, the function will load directly from the
        cached pickle object.
        If True and the pickle file does not exist, the function will load the CSV from
        disk and then save a pickle in the .eab_tools_cache dir with xz compression.
        If False, pickle objects will not be written or read.

    pkl_name : PathLike, Optional
        The filename to use for the cached pickle object if it should not inherit from
        the underlying datafile filename.

    kwargs
        Keyword arguments to be passed to `pandas.read_csv` or `pandas.read_excel`,
        depending on the datafile's filetype.

    Returns
    -------
    pandas.DataFrame
        The datafile as a pandas DataFrame object.

    Examples
    --------
    >>> iris_filepath = getfixture("iris_path")
    >>> df = load_df(iris_filepath)
    Attempting to load the file from disk...
    >>> df
         SepalLength  SepalWidth  PetalLength  PetalWidth            Name
    0            5.1         3.5          1.4         0.2     Iris-setosa
    1            4.9         3.0          1.4         0.2     Iris-setosa
    2            4.7         3.2          1.3         0.2     Iris-setosa
    3            4.6         3.1          1.5         0.2     Iris-setosa
    4            5.0         3.6          1.4         0.2     Iris-setosa
    ..           ...         ...          ...         ...             ...
    145          6.7         3.0          5.2         2.3  Iris-virginica
    146          6.3         2.5          5.0         1.9  Iris-virginica
    147          6.5         3.0          5.2         2.0  Iris-virginica
    148          6.2         3.4          5.4         2.3  Iris-virginica
    149          5.9         3.0          5.1         1.8  Iris-virginica
    <BLANKLINE>
    [150 rows x 5 columns]
    >>> df_again = load_df(iris_filepath)
    Loading from pickle...
    >>> (df == df_again).all(axis=None)
    True
    """
    CSV, XLS, XLSX = "csv", "xls", "xlsx"
    FILE_TYPES = pd.Series([CSV, XLS, XLSX])
    # Make sure it's a Path object
    filepath = Path(filepath)
    # Cleanup the file_type
    file_type_cf = file_type.casefold().replace(".", "")
    # Get the name of the file
    name = filepath.name

    # If loading an Excel sheet, add the sheetname to the
    # cache file's filename. This way, different sheets can
    # be read without overwriting each other's cache.
    if "sheet_name" in kwargs:
        name = f"{name} - {kwargs['sheet_name']}"

    # Determine the filetype or raise `ValueError`
    if file_type == "detect":
        suffixes = [suffix.casefold().replace(".", "") for suffix in filepath.suffixes]
        matches_mask = FILE_TYPES.isin(suffixes)
        matches = FILE_TYPES[matches_mask]

        if len(matches) == 1:
            file_type_cf = matches.iloc[0]
        else:
            file_type_cf = "".join(filepath.suffixes)
    if file_type_cf not in FILE_TYPES.values:
        raise ValueError(f"Could not parse file of type {file_type_cf}")

    # If cache is True, try to load from cache
    file_id = f"{name}{filepath.stat().st_mtime}"
    data_dir = filepath.resolve().parent
    pkl_name = pkl_name if pkl_name else file_id
    pkl_path = data_dir / ".eab_tools_cache" / f"{pkl_name}.pkl.xz"
    if cache and pkl_path.exists():
        print("Loading from pickle...")
        df = pd.read_pickle(pkl_path)

    # Otherwise, read the file from disk
    else:
        print("Attempting to load the file from disk...")
        if file_type_cf == "csv":
            df = pd.read_csv(filepath, **kwargs)
        elif file_type_cf == "xls":
            df = pd.read_excel(filepath, **kwargs)
        elif file_type_cf == "xlsx":
            engine = kwargs.pop("engine", "openpyxl")
            df = pd.read_excel(filepath, engine=engine, **kwargs)

    # Convert to `pandas` datatypes
    df = df.convert_dtypes(convert_boolean=False)

    if cache:
        # Pickle the df we have just loaded so it's faster
        # next time
        (data_dir / ".eab_tools_cache").mkdir(exist_ok=True)
        df.to_pickle(pkl_path)
    return df


def enrollments_report_date(
    filename: PathLike, encoding: str = "utf_8_sig"
) -> pd.Timestamp:
    """Detect the report date of an enrollments report."""
    # Grab the report date from one of the cells *just above* the header
    with open(filename, encoding=encoding) as f:
        date = f.readline().split(",")[3].replace('"', "")
        strftime = r"%m/%d/%Y %I:%M %p"
        return pd.to_datetime(date, format=strftime)
