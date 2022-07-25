from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd
import pytest

from EAB_tools import load_df


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


class TestLoadDf:
    iris_csv = Path(__file__).parent / "data" / "iris.csv"

    def test_doesnt_fail(self) -> None:
        load_df(self.iris_csv)

    def test_load_iris(self, iris: pd.DataFrame) -> None:
        df = load_df(self.iris_csv)

        assert (df == iris).all(axis=None)

    @pytest.mark.parametrize("file_type_specification", ["csv", "CSV", ".csv", ".CSV"])
    def test_specify_file_type_csv(
        self, file_type_specification: str, iris: pd.DataFrame
    ) -> None:
        # Make a copy of the csv with a weird extension
        weird_file = Path(str(self.iris_csv) + ".foo")
        shutil.copy(self.iris_csv, weird_file)

        df = load_df(weird_file, file_type=file_type_specification)
        assert (df == iris).all(axis=None)

        # Clean up
        weird_file.unlink()
