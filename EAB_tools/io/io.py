"""Methods to load data from disk."""

from typing import Optional

import pandas as pd

from .display import PathLike


def load_df(
    filepath: PathLike, file_type: str = "detect", pkl_name: Optional[PathLike] = None
) -> pd.DataFrame:
    """Load a CSV or Excel file as a pandas DataFrame."""
    pass
