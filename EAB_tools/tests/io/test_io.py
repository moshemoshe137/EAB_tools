"""
Tests the functionality of the data loading methods within the EAB_tools package.

This module includes tests for verifying the correct behavior of functions designed to
load data from various file types. It checks the functionality of caching mechanisms,
error handling with incorrect file types, and specific data consistency checks across
different file formats.
"""

from pathlib import Path
import shutil
from typing import ContextManager

from _pytest.capture import CaptureFixture
import pandas as pd
import pytest

from EAB_tools import load_df
from EAB_tools._testing import (
    PathLike,
    does_not_raise,
)

try:
    import xlrd  # noqa: F401 # 'xlrd' imported but unused

    _HAS_XLRD = True
except ImportError:
    _HAS_XLRD = False
try:
    import openpyxl  # noqa: F401 # 'openpyxl' imported but unused

    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False


def all_mixups(file_extension: str) -> list[str]:
    """
    Generate all variations of a file extension for testing case and dot sensitivity.

    Parameters
    ----------
    file_extension : str
        The file extension to test, with or without a leading dot.

    Returns
    -------
    list[str]
        A list of file extension variations.

    Examples
    --------
    # >>> all_mixups("csv")
    ['CSV', 'csv', '.CSV', '.csv']
    """
    if file_extension.startswith("."):
        file_extension = file_extension[1:]
    prefixes = ["", "."]
    funcs = [str.upper, str.lower]
    return [prefix + func(file_extension) for prefix in prefixes for func in funcs]


@pytest.mark.parametrize("cache", [True, False], ids="cache={}".format)
class TestLoadDf:
    """
    A collection of tests for the `load_df` function.

    The class tests the function's ability to handle different file formats, including
    Excel and CSV files, with emphasis on the function's response to cache usage,
    file naming variations, and incorrect file specifications.

    Methods
    -------
    test_doesnt_fail:
        Ensures that loading data does not fail for supported file types.
    test_load_iris:
        Compares loaded data against a reference DataFrame to ensure consistency.
    test_specify_file_type:
        Checks file type override capability when loading files with non-standard
        extensions.
    test_pickle_name:
        Verifies custom pickle file naming during the caching process.
    test_multiple_excel_sheets:
        Tests loading specific or multiple sheets from Excel files.
    test_bad_filetype:
        Asserts that loading fails with an appropriate error messages for unsupported
        file types.
    test_ambiguous_filetype:
        Examines error handling when file extensions are ambiguous.
    test_wrong_filetype:
        Ensures that loading fails correctly when the wrong file type is specified.
    test_load_df_cache:
        Tests caching functionality, verifying that files are loaded from cache
        correctly.
    """

    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("iris*"))

    # Mark files with `pytest` `param`s
    files_with_marks = []
    for file in files:
        if "xlsx" in file.suffix.casefold():
            param = pytest.param(
                file,
                marks=pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl required"),
            )
        elif "xls" in file.suffix.casefold():
            param = pytest.param(
                file, marks=pytest.mark.skipif(not _HAS_XLRD, reason="xlrd required")
            )
        else:
            param = pytest.param(file)
        files_with_marks.append(param)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_doesnt_fail(self, cache: bool, file: PathLike, tmp_path: Path) -> None:
        """
        Ensure that loading a file does not raise an exception.

        Parameters
        ----------
        cache : bool
            Whether to enable caching for the loading process.
        file : PathLike
            The file path to be tested.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        load_df(file, cache=cache, cache_dir=tmp_path)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_load_iris(
        self, file: PathLike, iris: pd.DataFrame, cache: bool, tmp_path: Path
    ) -> None:
        """
        Confirm the DataFrame loaded from the file matches a known DataFrame, Iris.

        Parameters
        ----------
        file : PathLike
            The file path to be tested.
        iris : pd.DataFrame
            The Iris reference DataFrame to compare against.
        cache : bool
            Whether to use caching.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        df = load_df(file, cache=cache, cache_dir=tmp_path)

        assert (df == iris).all(axis=None)

    # For tests that need both the file and its potential marks
    @pytest.mark.parametrize(
        "file, file_type_specification",
        [
            pytest.param(file, suffix_specification, marks=file_with_marks.marks)
            for file, file_with_marks in zip(files, files_with_marks)
            for suffix_specification in all_mixups(file.suffix)
        ],
        ids=str,
    )
    def test_specify_file_type(
        self,
        file: PathLike,
        file_type_specification: str,
        iris: pd.DataFrame,
        cache: bool,
        tmp_path: Path,
    ) -> None:
        """
        Test file loading with specific file type specification.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        file_type_specification : str
            The file type specification to test.
        iris : pd.DataFrame
            The reference DataFrame to compare against.
        cache : bool
            Whether to enable caching for the loading process.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        # Make a copy of the csv with a weird extension
        weird_file = tmp_path / Path(str(file) + ".foo").name
        if not weird_file.exists():
            shutil.copy(file, weird_file)

        df = load_df(
            weird_file,
            file_type=file_type_specification,
            cache=cache,
            cache_dir=tmp_path,
        )
        assert (df == iris).all(axis=None)

        # Clean up
        weird_file.unlink()

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    @pytest.mark.parametrize(
        "pkl_name",
        [
            None,
            "foo",
            "baz.bar",
            Path("oof"),
            Path("rab.zab"),
            "test.bz2",
            Path("test_path.bz2"),
        ],
    )
    def test_pickle_name(
        self, file: PathLike, cache: bool, pkl_name: PathLike, tmp_path: Path
    ) -> None:
        """
        Test the naming and creation of cache files during data loading.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        cache : bool
            Indicates if caching is enabled (must be True for this test).
        pkl_name : PathLike
            The custom name for the pickle file, if specified.
        tmp_path : Path
            Temporary directory path used for caching.
        """
        load_df(
            file,
            cache=True,  # Must be true to test pickling
            cache_dir=tmp_path,
            pkl_name=pkl_name,
        )
        if pkl_name is None:
            pkl_name = f"{file.name}{file.stat().st_mtime}.pkl.xz"
        else:
            pkl_name = f"{pkl_name}.pkl.xz"

        assert (tmp_path / pkl_name).exists()

    @pytest.mark.parametrize("sn", ["spam", "eggs", "spam&eggs", "iris", None])
    @pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl required")
    def test_multiple_excel_sheets(self, cache: bool, sn: str, tmp_path: Path) -> None:
        """
        Test loading specific sheets from an Excel file with multiple sheets.

        Parameters
        ----------
        cache : bool
            Whether to enable caching for the loading process.
        sn : str
            The name of the sheet to load. None implies all sheets should be loaded.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        file = self.data_dir / "multiple_sheets.xlsx"

        # `sheet_name = None` behavior is not yet defined and right now is expected
        # to just raise an exception
        context = does_not_raise() if sn else pytest.raises(Exception)

        # Needed to make mypy happy
        assert isinstance(context, ContextManager)

        with context:
            load_df(file, cache=cache, cache_dir=tmp_path, sheet_name=sn)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    @pytest.mark.parametrize("bad_file_type", [".db", "gsheets", "exe", ".PY"])
    def test_bad_filetype(
        self, file: PathLike, cache: bool, bad_file_type: str, tmp_path: Path
    ) -> None:
        """
        Test file loading with unsupported file types to confirm proper error handling.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        cache : bool
            Whether to enable caching for the loading process.
        bad_file_type : str
            The unsupported file type used to trigger the exception.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        msg = "Could not parse file of type"
        with pytest.raises(ValueError, match=msg):
            load_df(file, file_type=bad_file_type, cache=cache, cache_dir=tmp_path)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    @pytest.mark.parametrize("ambiguous_suffix", [".csv.xlsx", ".xlsx.csv"])
    def test_ambiguous_filetype(
        self, file: PathLike, cache: bool, ambiguous_suffix: str, tmp_path: Path
    ) -> None:
        """
        Test handling of files with ambiguous file extensions.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        cache : bool
            Whether to enable caching for the loading process.
        ambiguous_suffix : str
            The ambiguous suffix to append to the file name for testing.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        file = Path(file)  # Make mypy happy
        msg = r"Ambiguous suffix\(es\):"
        with pytest.raises(ValueError, match=msg):
            load_df(
                f"{file.name}{ambiguous_suffix}",
                file_type="detect",
                cache=cache,
                cache_dir=tmp_path,
            )

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_wrong_filetype(self, file: PathLike, cache: bool, tmp_path: Path) -> None:
        """
        Test loading files with incorrect file type specifications raises an exception.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        cache : bool
            Whether to enable caching for the loading process.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        file = Path(file)
        my_file_type = file.suffix.casefold().replace(".", "")
        wrong_file_types = [
            suffix for suffix in ["csv", "xls", "xlsx"] if suffix not in my_file_type
        ]
        if not _HAS_XLRD and "xls" in wrong_file_types:
            # `xlrd` is required for .xls files
            wrong_file_types.remove("xls")
        if not _HAS_OPENPYXL and "xlsx" in wrong_file_types:
            # `openpyxl` is required for .xlsx files
            wrong_file_types.remove("xlsx")

        msg = r"""(?xi)
        can't\ decode\ byte  # UnicodeDecodeError
        | Excel\ file\ format\ cannot\ be\ determined  # ValueError
        | no\ valid\ workbook  # OSError
        | File\ is\ not\ a\ zip\ file  # zipfile.BadZipFile
        """
        for wrong_file_type in wrong_file_types:
            with pytest.raises(Exception, match=msg):
                load_df(
                    file, file_type=wrong_file_type, cache=cache, cache_dir=tmp_path
                )

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_load_df_cache(
        self,
        file: PathLike,
        cache: bool,
        capsys: CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """
        Test the functionality of caching in data loading.

        Parameters
        ----------
        file : PathLike
            The file path being tested.
        cache : bool
            Whether to enable caching for the loading process.
        capsys : CaptureFixture[str]
            `pytest` fixture to capture print statements.
        tmp_path : Path
            Temporary directory path used for caching, if enabled.
        """
        file = Path(file)  # Make mypy happy

        # Make sure one read comes from the file
        df = load_df(file, cache=True, cache_dir=tmp_path)
        assert "Attempting to load the file from disk..." in capsys.readouterr().out

        # Make sure the other read comes from the pickle
        df_cached = load_df(file, cache=True, cache_dir=tmp_path)
        assert "Loading from pickle..." in capsys.readouterr().out

        # They better be equal!
        assert (df == df_cached).all(axis=None)
