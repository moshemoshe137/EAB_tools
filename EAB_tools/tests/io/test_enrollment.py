from pathlib import Path
import shutil

import pandas as pd
import pytest

from EAB_tools import (
    load_all_enrollment_reports,
    load_enrollments_report,
)
from EAB_tools.io.enrollment import enrollments_report_date


def test_enrollments_report_fixture(enrollments_report: pd.DataFrame) -> None:
    assert isinstance(enrollments_report, pd.DataFrame)

    assert "Student ID" in enrollments_report.columns

    assert len(enrollments_report) > 5


def test_enrollments_report_unique_ids(enrollments_report: pd.DataFrame) -> None:
    # Every name is assigned to one unique ID
    assert (
        enrollments_report.groupby("Student ID")["Student Name"].nunique() == 1
    ).all()

    # Some ID numbers show up more than once
    assert (enrollments_report["Student ID"].value_counts() > 1).any()


def test_enrollments_report_emails(enrollments_report: pd.DataFrame) -> None:
    assert (
        enrollments_report.groupby("Student E-mail")["Student Name"].nunique() == 1
    ).all()

    emails_without_dots = enrollments_report["Student E-mail"].str.replace(".", "")
    assert (
        enrollments_report.groupby(emails_without_dots)["Student ID"].nunique() == 1
    ).all()


def test_different_enrollments_report(
    different_enrollments_report: Path, enrollments_report: pd.DataFrame
) -> None:
    different_report = load_enrollments_report(different_enrollments_report)
    n = len(different_report)
    enrollments_report_head = enrollments_report[:n]

    assert (
        different_report["Student ID"] != enrollments_report_head["Student ID"]
    ).all()

    assert (
        different_report["Report Date"] == pd.to_datetime("03/10/2024 20:30:00")
    ).all()


def test_enrollments_report_date(
    different_enrollments_report: Path,
    enrollments_report: pd.DataFrame,
    enrollments_report_path: Path,
) -> None:
    different_date = pd.to_datetime("03/10/2024 20:30:00")
    assert enrollments_report_date(different_enrollments_report) == different_date

    enrollments_df = load_enrollments_report(enrollments_report_path)
    assert (
        enrollments_report_date(enrollments_report_path)
        == enrollments_df["Report Date"]
    ).all()


@pytest.mark.parametrize("ignore_index", [True, False])
class TestLoadAllEnrollmentReports:
    @pytest.mark.slow
    def test_load_all_enrollment_reports(
        self, data_dir: Path, ignore_index: bool, tmp_path: Path
    ) -> None:
        load_all_enrollment_reports(
            data_dir, ignore_index=ignore_index, cache_dir=tmp_path
        )

    def test_load_all_enrollments_quicker(
        self, data_dir: Path, tmp_path: Path, ignore_index: bool
    ) -> None:
        for file in data_dir.glob("*.csv"):
            if file.stat().st_size < 50 * 1024 * 1024:  # 50 MB
                shutil.copy2(file, tmp_path)

        load_all_enrollment_reports(
            tmp_path, ignore_index=ignore_index, cache_dir=tmp_path
        )
