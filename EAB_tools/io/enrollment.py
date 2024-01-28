"""Load EAB enrollment reports."""

from typing import Optional

import pandas as pd

from EAB_tools import load_df
import EAB_tools._testing as tm

from .io import enrollments_report_date


def load_enrollments_report(
    filename: tm.PathLike,
    cache: bool = True,
    pkl_name: Optional[tm.PathLike] = None,
    convert_dates: bool = True,
    convert_section_to_string: bool = True,
    convert_categoricals: bool = True,
) -> pd.DataFrame:
    """Load and EAB enrollments report and add a column with the report date."""
    df = load_df(filename, cache=cache, pkl_name=pkl_name)
    df["Report Date"] = enrollments_report_date(filename)

    if convert_dates:
        # Convert the remaining datetime fields to appropriate dtype
        date_fields = [
            "Dropped Date",
            "Start Date",
            "End Date",
            "Start Time",
            "End Time",
        ]
        df[date_fields] = pd.to_datetime(df[date_fields])

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
