from pathlib import Path
import shutil
from typing import Iterator

import pandas as pd
import pytest

from EAB_tools import load_df


@pytest.fixture(autouse=True)
def _test_cache_cleanup() -> Iterator[None]:
    # Before the test
    pass

    # yield
    yield

    # After the test
    cache_path = Path(__file__).parent / "data" / ".eab_tools_cache"
    shutil.rmtree(cache_path)


class TestLoadDf:
    iris_csv = Path(__file__).parent / "data" / "iris.csv"

    def test_doesnt_fail(self) -> None:
        load_df(self.iris_csv)

    def test_load_iris(self, iris: pd.DataFrame) -> None:
        df = load_df(self.iris_csv)

        assert (df == iris).all(axis=None)
