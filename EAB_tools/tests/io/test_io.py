from pathlib import Path

from EAB_tools import load_df


class TestLoadDf:
    iris_csv = Path(__file__).parent / "data / iris.csv"

    def test_doesnt_fail(self) -> None:
        load_df(self.iris_csv)
