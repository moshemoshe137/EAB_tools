from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd
import pytest

from EAB_tools import load_df
import EAB_tools._testing as tm


@pytest.fixture(autouse=True)
def _test_cache_cleanup() -> Iterator[None]:
    # Before the test
    cache_path = Path(__file__).parent / "data" / ".eab_tools_cache"
    cache_path.mkdir(exist_ok=True)
    pass

    # yield
    yield

    # After the test
    shutil.rmtree(cache_path)


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
    >>> all_mixups("csv")
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

    @pytest.mark.parametrize("file", files, ids=lambda pth: pth.name)
    def test_doesnt_fail(self, cache: bool, file: tm.PathLike) -> None:
        load_df(file, cache=cache)

    @pytest.mark.parametrize("file", files, ids=lambda pth: pth.name)
    def test_load_iris(
        self, file: tm.PathLike, iris: pd.DataFrame, cache: bool
    ) -> None:
        df = load_df(file, cache=cache)

        assert (df == iris).all(axis=None)

    @pytest.mark.parametrize(
        "file,file_type_specification",
        [
            (file, suffix_specification)
            for file in files
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

    @pytest.mark.parametrize("file", files, ids=lambda pth: pth.name)
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
