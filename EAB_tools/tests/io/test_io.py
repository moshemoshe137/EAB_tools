from pathlib import Path
import shutil
from typing import ContextManager

from _pytest.capture import CaptureFixture
import pandas as pd
import pytest

from EAB_tools import load_df
import EAB_tools._testing as tm

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
    Get all the mix-ups we're looking to test.

    Parameters
    ----------
    file_extension : str
        A file extension

    Returns
    -------
    list[str]
        All mixups being considered

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
    def test_doesnt_fail(self, cache: bool, file: tm.PathLike) -> None:
        load_df(file, cache=cache)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_load_iris(
        self, file: tm.PathLike, iris: pd.DataFrame, cache: bool
    ) -> None:
        df = load_df(file, cache=cache)

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
        file: tm.PathLike,
        file_type_specification: str,
        iris: pd.DataFrame,
        cache: bool,
    ) -> None:
        # Make a copy of the csv with a weird extension
        weird_file = Path(str(file) + ".foo")
        shutil.copy(file, weird_file)

        df = load_df(weird_file, file_type=file_type_specification, cache=cache)
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
        self, file: tm.PathLike, cache: bool, pkl_name: tm.PathLike
    ) -> None:
        load_df(
            file,
            cache=True,  # Must be true to test pickling
            pkl_name=pkl_name,
        )
        if pkl_name is None:
            pkl_name = f"{file.name}{file.stat().st_mtime}.pkl.xz"
        else:
            pkl_name = f"{pkl_name}.pkl.xz"

        assert (Path(file).parent / ".eab_tools_cache" / pkl_name).exists()

    @pytest.mark.parametrize("sn", ["spam", "eggs", "spam&eggs", "iris", None])
    @pytest.mark.skipif(not _HAS_OPENPYXL, reason="openpyxl required")
    def test_multiple_excel_sheets(self, cache: bool, sn: str) -> None:
        file = self.data_dir / "multiple_sheets.xlsx"

        # `sheet_name = None` behavior is not yet defined and right now is expected
        # to just raise an exception
        context = tm.does_not_raise() if sn else pytest.raises(Exception)

        # Needed to make mypy happy
        assert isinstance(context, ContextManager)

        with context:
            load_df(file, cache=cache, sheet_name=sn)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    @pytest.mark.parametrize("bad_file_type", [".db", "gsheets", "exe", ".PY"])
    def test_bad_filetype(
        self, file: tm.PathLike, cache: bool, bad_file_type: str
    ) -> None:
        msg = "Could not parse file of type"
        with pytest.raises(ValueError, match=msg):
            load_df(file, cache=cache, file_type=bad_file_type)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    @pytest.mark.parametrize("ambiguous_suffix", [".csv.xlsx", ".xlsx.csv"])
    def test_ambiguous_filetype(
        self, file: tm.PathLike, cache: bool, ambiguous_suffix: str
    ) -> None:
        file = Path(file)  # Make mypy happy
        msg = r"Ambiguous suffix\(es\):"
        with pytest.raises(ValueError, match=msg):
            load_df(f"{file.name}{ambiguous_suffix}", cache=cache, file_type="detect")

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_wrong_filetype(self, file: tm.PathLike, cache: bool) -> None:
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
                load_df(file, cache=cache, file_type=wrong_file_type)

    @pytest.mark.parametrize("file", files_with_marks, ids=lambda pth: pth.name)
    def test_load_df_cache(
        self, file: tm.PathLike, cache: bool, capsys: CaptureFixture[str]
    ) -> None:
        file = Path(file)  # Make mypy happy

        # Make sure one read comes from the file
        df = load_df(file, cache=True)
        assert "Attempting to load the file from disk..." in capsys.readouterr().out

        # Make sure the other read comes from the pickle
        df_cached = load_df(file, cache=True)
        assert "Loading from pickle..." in capsys.readouterr().out

        # They better be equal!
        assert (df == df_cached).all(axis=None)
