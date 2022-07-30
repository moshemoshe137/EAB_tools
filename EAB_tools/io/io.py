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
    """Load a CSV or Excel file as a pandas DataFrame."""
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
