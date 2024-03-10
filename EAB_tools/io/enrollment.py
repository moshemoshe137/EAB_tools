"""Load EAB enrollment reports."""

from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    Optional,
)

import pandas as pd

from EAB_tools import load_df
import EAB_tools._testing as tm

from .display import PathLike


def enrollments_report_date(
    filename: PathLike, encoding: str = "utf_8_sig"
) -> pd.Timestamp:
    """Detect the report date of an enrollments report."""
    # Grab the report date from one of the cells *just above* the header
    with open(filename, encoding=encoding) as f:
        date = f.readline().split(",")[3].replace('"', "")
        strftime = r"%m/%d/%Y %H:%M:%S"
        return pd.to_datetime(date, format=strftime)


def load_enrollments_report(
    filename: tm.PathLike,
    cache: bool = True,
    pkl_name: Optional[tm.PathLike] = None,
    convert_dates: bool = True,
    convert_section_to_string: bool = True,
    convert_categoricals: bool = True,
    header: int | Sequence[int] | None | Literal["infer"] = 1,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load and EAB enrollments report and add a column with the report date."""
    df = load_df(filename, cache=cache, pkl_name=pkl_name, header=header, **kwargs)
    df["Report Date"] = enrollments_report_date(filename)

    if convert_dates:
        # Convert the remaining datetime fields to appropriate dtype
        date_fields, time_fields = [
            "Dropped Date",
            "Start Date",
            "End Date",
        ], ["Start Time", "End Time"]
        for date_field in date_fields:
            df[date_field] = pd.to_datetime(df[date_field])
        for time_field in time_fields:
            suffix = df[time_field].str.extract(r".+ (?:AM|PM)?( [A-z]+)$").iloc[0, 0]
            if not pd.isna(suffix):
                # Remove the suffix
                df[time_field] = df[time_field].str.replace(suffix, "")
            df[time_field] = pd.to_datetime(df[time_field], format="%I:%M %p")

    if convert_section_to_string:
        # Convert any other fields to appropriate dtypes
        df["Section"] = df.Section.astype("string")

    if convert_categoricals:
        categoricals = [
            "Major",
            "Course Name",
            "Course Number",
            "Section",
            "Instructors",
            "Dropped?",
            "Midterm Grade",
            "Final Grade",
            "Class Days",
        ]
        df[categoricals] = df[categoricals].astype("category")

    return df
