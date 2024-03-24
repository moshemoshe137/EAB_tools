from collections.abc import (
    Iterable,
    Iterator,
)
import itertools
import os
from pathlib import Path
import shutil
import subprocess
from typing import (
    Any,
    Callable,
    Literal,
    Union,
)

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from EAB_tools._testing.types import PytestFixtureRequest
import EAB_tools.tests.io.data.generate_all_test_data

Numeric = np.number[Any]

_iris_Path = Path(__file__).parent / "tests" / "io" / "data" / "iris.csv"
iris_df: pd.DataFrame = pd.read_csv(_iris_Path)

_enrollments_Path = (
    Path(__file__).parent / "tests" / "io" / "data" / "campus-v2report-enrollment.csv"
)

_generate_enrollments_path = (
    Path(__file__) / "tests" / "io" / "data" / "generate_fake_enrollments.py"
)

# Set up the type for mypy
enrollments_df: pd.DataFrame


@pytest.fixture(autouse=True, scope="session")
def generate_all_test_data() -> None:
    EAB_tools.tests.io.data.generate_all_test_data.main()
    global enrollments_df
    enrollments_df = pd.read_csv(_enrollments_Path, header=1)


@pytest.fixture
def iris_path() -> Path:
    """pathlib Path object of absolute path to iris CSV file."""
    return _iris_Path.resolve()


@pytest.fixture
def iris() -> pd.DataFrame:
    """The iris dataset as a pandas DataFrame."""
    return iris_df


@pytest.fixture(params=iris_df.columns)
def iris_cols(request: PytestFixtureRequest) -> pd.Series:
    """Return iris dataframe columns, one after the next"""
    return iris_df[request.param]


@pytest.fixture(
    params=[str, pytest.param(lambda col_name: pd.Index([col_name]), id="pd.Index")]
)
def iris_single_col_subset(
    iris_cols: pd.Series, request: PytestFixtureRequest
) -> Union[str, pd.Index]:
    """Return a col name as a str or pd.Index"""
    func = request.param
    col = iris_cols.name
    return func(col)


@pytest.fixture
def enrollments_report_path() -> Path:
    return _enrollments_Path.resolve()


@pytest.fixture
def enrollments_report() -> pd.DataFrame:
    return enrollments_df


@pytest.fixture
def generate_new_enrollments_report() -> pd.DataFrame:
    global enrollments_df
    subprocess.run(["ipython", str(_generate_enrollments_path)])
    enrollments_df = pd.read_csv(_enrollments_Path, header=1)
    return enrollments_df


@pytest.fixture
def different_enrollments_report(tmp_path: Path) -> pd.DataFrame:
    sample_enrollments_report = """Example University,Student Enrollments,,03/10/2024 20:30:00,Moshe Rubin

Student Name,Student E-mail,Student ID,Student Alternate ID,Categories,Tags,Classification,Major,Cumulative GPA,Assigned Staff,Course Name,Course Number,Section,Instructors,Dropped?,Dropped Date,Midterm Grade,Final Grade,Total Progress Reports,Absences,Unexcused Absences,Excused Absences,Credit Hours,Start Date,End Date,Start Time,End Time,Class Days
"Hudson, Susan",susan.hudson@example.edu,ID232237800,,"Graduated: No, Undergraduate, Completion Rate: >= 66.67%, Start Term: Fall 2021, Term Status: Registered, FAFSA: No",,Graduate (Winter 2024),Journalism,0.0,"Curtis Evans (Advisor), Donald Roberts (Career Advisor), Joshua King (Professor), Tina Pena (Professor)",Customer dream pattern police leave,MED-371,56477,"Grant, Monique (ID738164466) <monique.grant@example.edu>; King, Joshua (ID509560200) <joshua.king2@example.edu>",Yes,2023-05-19,B,F,0,5,5,0,5,2024-04-25,2024-07-04,10:30 AM CT,2:30 PM CT,TR
"Hudson, Susan",susan.hudson@example.edu,ID232237800,,"Graduated: No, Undergraduate, Completion Rate: >= 66.67%, Start Term: Fall 2021, Term Status: Registered, FAFSA: No",,Graduate (Winter 2024),Journalism,0.0,"Curtis Evans (Advisor), Donald Roberts (Career Advisor), Joshua King (Professor), Tina Pena (Professor)",Truth indicate wait act nearly form,STT-557,10891,"Pena, Tina (ID230553056) <tina.pena@example.edu>",Yes,2024-03-02,D,A,1,3,3,0,5,2024-01-16,2024-03-26,1:00 PM CT,2:15 PM CT,MW
"Hudson, Susan",susan.hudson@example.edu,ID232237800,,"Graduated: No, Undergraduate, Completion Rate: >= 66.67%, Start Term: Fall 2021, Term Status: Registered, FAFSA: No",,Graduate (Winter 2024),Journalism,0.0,"Curtis Evans (Advisor), Donald Roberts (Career Advisor), Joshua King (Professor), Tina Pena (Professor)",Success operation television treat together change,ENG-389,99535,"Gumprich, Gustav (ID420086633) <gustav.gumprich@example.edu>",Yes,2023-09-20,D,C,1,2,2,0,5,2024-06-05,2024-08-14,9:00 AM CT,12:00 PM CT,W
"Huff, Amanda",amanda.huff@example.edu,ID252221822,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Fall 2023, Term Status: Registered, FAFSA: No",,,English Literature,0.32,"Brandi Hurley (Professor), Frank Rivera (Advisor), Jason Navarro (FAFSA coordinator), Nicole Bennett (Career Advisor)",Us cut job discuss garden activity force safe,OPH-400,35682,"Hurley, Brandi (ID332794299) <brandi.hurley@example.edu>",Yes,2023-04-25,A,D,0,2,2,0,5,2024-03-22,2024-05-31,2:00 PM CT,5:00 PM CT,T
"Huff, Amanda",amanda.huff@example.edu,ID252221822,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Fall 2023, Term Status: Registered, FAFSA: No",,,English Literature,0.32,"Brandi Hurley (Professor), Frank Rivera (Advisor), Jason Navarro (FAFSA coordinator), Nicole Bennett (Career Advisor)",Spend these short,ART-595,38939,"Meunier, Hortense (ID652184974) <hortense.meunier@example.edu>",No,,B,D,0,5,5,0,5,2024-02-03,2024-04-13,9:00 AM CT,10:15 AM CT,MWF
"Douglas, Tracey",tracey.douglas@example.edu,ID711775411,,"Main Campus, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Spring 2022, Term Status: Registered, FAFSA: No",,Foo (Winter 2024),Graphic Design,3.28,"Randall Morgan (Career Advisor), Ryan Stone (FAFSA coordinator), Scott Williams (Advisor)",Exactly line friend word only,URB-175,35709,"Clark, Jessica (ID505531377) <jessica.clark2@example.edu>",Yes,2023-07-01,D,A,0,0,2,0,3,2024-05-30,2024-08-08,9:00 AM CT,12:00 PM CT,TR
"Douglas, Tracey",tracey.douglas@example.edu,ID711775411,,"Main Campus, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Spring 2022, Term Status: Registered, FAFSA: No",,Foo (Winter 2024),Graphic Design,3.28,"Randall Morgan (Career Advisor), Ryan Stone (FAFSA coordinator), Scott Williams (Advisor)",Bed close let see today case,PTH-132,76443,"Smith, Tammy (ID710839343) <tammy.smith2@example.edu>",Yes,2023-12-07,A,C,1,6,6,0,3,2024-03-01,2024-05-10,3:00 PM CT,9:00 PM CT,MW
"Douglas, Tracey",tracey.douglas@example.edu,ID711775411,,"Main Campus, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Spring 2022, Term Status: Registered, FAFSA: No",,Foo (Winter 2024),Graphic Design,3.28,"Randall Morgan (Career Advisor), Ryan Stone (FAFSA coordinator), Scott Williams (Advisor)",Feeling room half sing improve,URB-333,80092,"Christian, Andrew (ID080723550) <andrew.christian@example.edu>",No,,B,D,0,3,3,0,3,2024-04-21,2024-06-30,1:00 PM CT,4:00 PM CT,F
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Second less hit charge his,SUR-300,33707,"Robertson, Shawn (ID650374707) <shawn.robertson@example.edu>",No,,A,C,2,2,2,0,12,2024-04-01,2024-06-10,1:00 PM CT,2:15 PM CT,W
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Wall later believe knowledge discover bill,MUS-121,75814,"Jenkins, Diana (ID436548473) <diana.jenkins@example.edu>",Yes,2023-09-28,A,C,1,5,5,0,12,2023-12-23,2024-03-02,9:00 AM CT,10:15 AM CT,MWF
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Heavy leg six magazine course far who,ROM-119,35138,"Dodson, Kyle (ID858069184) <kyle.dodson@example.edu>",No,,B,A,2,4,4,0,12,2024-05-15,2024-07-24,4:00 PM CT,7:00 PM CT,
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Allow modern film defense,LAN-441,22765,"Kirby, Julian (ID385836169) <julian.kirby@example.edu>",No,,F,D,0,3,3,0,12,2024-04-04,2024-06-13,4:30 PM CT,5:45 PM CT,TR
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Republican short first war value,URB-192,53853,"Coffey, Brenda (ID798122417) <brenda.coffey@example.edu>",Yes,2023-08-16,D,C,0,3,3,0,12,2024-03-22,2024-05-31,1:00 PM CT,4:00 PM CT,MWF
"Collins, Catherine",catherine.collins@example.edu,ID303476034,,"On-line, Graduated: No, Graduate, Completion Rate: >= 66.67%, Start Term: Winter 2016, Term Status: Registered, FAFSA: No",At Risk,Graduate (Winter 2024),Biochemistry,0.0,Timothy Coleman (Career Advisor),Father impact woman outside example ever father,AHS-156,90666,"Waters, Emily (ID758574272) <emily.waters@example.edu>",No,,B,A,0,3,3,0,12,2023-12-22,2024-03-01,12:00 PM CT,1:15 PM CT,
"Rivera, Donna",donna.rivera@example.edu,ID051164965,,"Satellite Campus, Graduated: No, Graduate, Completion Rate: < 66.67%, Start Term: Spring 2015, Term Status: Registered, FAFSA: No",,Foo (Spring 2024),Mechanical Engineering,4.0,"Amy Gonzalez (Career Advisor), Patricia Shaw (Professor)",Body official mind black possible,MTH-406,37113,"Sellers, Stephen (ID411073579) <stephen.sellers@example.edu>",No,,B,B,0,0,5,0,5,2024-02-03,2024-04-13,5:00 PM CT,9:00 PM CT,M
"Rivera, Donna",donna.rivera@example.edu,ID051164965,,"Satellite Campus, Graduated: No, Graduate, Completion Rate: < 66.67%, Start Term: Spring 2015, Term Status: Registered, FAFSA: No",,Foo (Spring 2024),Mechanical Engineering,4.0,"Amy Gonzalez (Career Advisor), Patricia Shaw (Professor)",Force half state somebody,THR-098,60727,"Smith, Cynthia (ID519409430) <cynthia.smith2@example.edu>",No,,B,A,0,0,3,0,5,2024-01-28,2024-04-07,1:00 PM CT,2:00 PM CT,MWF
"Rivera, Donna",donna.rivera@example.edu,ID051164965,,"Satellite Campus, Graduated: No, Graduate, Completion Rate: < 66.67%, Start Term: Spring 2015, Term Status: Registered, FAFSA: No",,Foo (Spring 2024),Mechanical Engineering,4.0,"Amy Gonzalez (Career Advisor), Patricia Shaw (Professor)",Pay media radio despite say cut my personal,ANT-144,86071,"Shaw, Patricia (ID740449289) <patricia.shaw@example.edu>",Yes,2023-04-02,C,F,0,1,1,0,5,2024-04-27,2024-07-06,4:30 PM CT,7:30 PM CT,MW
"""  # noqa: E501 (line too long)
    csv_path = tmp_path / "different_enrollments_report.csv"
    with open(csv_path, "w") as f:
        f.write(sample_enrollments_report)
    return csv_path


@pytest.fixture(
    params=[
        pytest.param(pd.Series([1, 2, 3] * 3, dtype="int32"), id="int32series"),
        pytest.param(
            pd.Series([None, 2.5, 3.5] * 3, dtype="float32"), id="float32series"
        ),
        pytest.param(
            pd.Series(["a", "b", "c"] * 3, dtype="category"), id="category_series"
        ),
        pytest.param(pd.Series(["d", "e", "f"] * 3), id="object_series"),
        pytest.param(pd.Series([True, False, True] * 3), id="bool_series"),
        pytest.param(
            pd.Series(pd.date_range("20130101", periods=9)), id="datetime_series"
        ),
        pytest.param(
            pd.Series(pd.date_range("20130101", periods=9, tz="US/Eastern")),
            id="datetime_tz_series",
        ),
        pytest.param(
            pd.Series(pd.timedelta_range("2000", periods=9)), id="timedelta_series"
        ),
    ]
)
def series(request: PytestFixtureRequest) -> pd.Series:
    """Return several series with unique dtypes"""
    # Fixture borrowed from pandas from
    # https://github.com/pandas-dev/pandas/blob/5b2fb093f6abd6f5022fe5459af8327c216c5808/pandas/tests/util/test_hashing.py
    return request.param


pairs = list(itertools.permutations(iris_df.columns, 2))


@pytest.fixture(params=pairs, ids=list(map(str, pairs)))
def multiindex(iris: pd.DataFrame, request: PytestFixtureRequest) -> pd.MultiIndex:
    """Return MultiIndexes created from pairs of iris cols"""
    a_col, b_col = request.param
    a, b = iris[a_col], iris[b_col]
    return pd.MultiIndex.from_arrays([a, b])


@pytest.fixture(
    params=[
        np.sin,  # any -> float
        pytest.param(lambda arr: np.exp(-arr), id="exp(-x)"),  # any -> float
        pytest.param(
            lambda x: x**2, id="lambda squared"
        ),  # int -> int and float -> float
        pytest.param(lambda arr: np.rint(arr).astype("int64"), id="rint"),  # any -> int
    ],
    name="func",
)
def plot_func(
    request: PytestFixtureRequest,
) -> Callable[[npt.ArrayLike], npt.NDArray[Numeric]]:
    """A variety of mathematical funcs callable on numeric numpy ndarrays"""
    return request.param


@pytest.fixture(
    params=[
        np.linspace(0, 10**-5, dtype=float),
        np.linspace(0, 499, num=500, dtype="int32"),
        np.linspace(0, 2**33, 2**10 + 1, dtype="int64"),
    ],
    ids=lambda arr: str(arr.dtype),
)
def x_values(request: PytestFixtureRequest) -> npt.NDArray[Numeric]:
    """func inputs of different dtypes"""
    return request.param


@pytest.fixture(params=["fig", "ax"])
def _fig_or_ax(request: PytestFixtureRequest) -> Literal["fig", "ax"]:
    """Either returns 'fig' or 'ax'"""
    return request.param


@pytest.fixture
def mpl_plots(
    func: Callable[[npt.ArrayLike], npt.NDArray[Numeric]],
    x_values: npt.NDArray[Numeric],
) -> Iterable[dict[str, Union[plt.Figure, plt.Axes]]]:
    """Returns dict of {fix, ax}, for various funcs and domains"""
    x = x_values
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)

    yield {"fig": fig, "ax": ax}
    plt.close(fig)


@pytest.fixture
def mpl_axes(mpl_plots: dict[str, Union[plt.Figure, plt.Axes]]) -> plt.Axes:
    """Returns a variety of `plt.Axes` objects"""
    assert isinstance(mpl_plots["ax"], plt.Axes)  # for mypy
    return mpl_plots["ax"]


@pytest.fixture
def mpl_figs(mpl_plots: dict[str, Union[plt.Figure, plt.Axes]]) -> plt.Figure:
    """Returns a variety of `plt.Figure` objects"""
    assert isinstance(mpl_plots["fig"], plt.Figure)  # for mypy
    return mpl_plots["fig"]


@pytest.fixture
def mpl_figs_and_axes(
    mpl_plots: dict[str, Union[plt.Figure, plt.Axes]], _fig_or_ax: Literal["fig", "ax"]
) -> Union[plt.Figure, plt.Axes]:
    """Returns either the figure or the axis of various plots"""
    return mpl_plots[_fig_or_ax]


today = pd.Timestamp.today(dateutil.tz.tzlocal())
strftime_codes = [
    "%B %d, %Y",  # June 12, 2022
    "%Y-%m-%d",  # 2022-06-12
    "%B %e, %Y",  # June 12, 2022
    "%a, %b %e",  # Sun, Jun 12
    "%e %b %Y",  # 12 Jun 2022
    "%A, %B %e, %Y",  # Sunday, June 12, 2022
    "%H:%M:%S",  # 16:51:45
    "%Y-%m-%dT%H:%M:%S.%f%z",  # 2022-06-12T16:51:45.576846-0500
    "%I:%M %p",  # 04:51 PM
]


@pytest.fixture(
    params=iter(strftime_codes),
    ids=strftime_codes,
)  # noqa
def strftime(request: PytestFixtureRequest) -> str:
    """Various different strftime format codes"""
    return request.param


@pytest.fixture(
    params=[
        pd.to_timedelta(0),
        pd.Timedelta(hours=15),
        pd.DataFrame(
            (pd.to_timedelta(row, unit="hours") for row in np.random.rand(10, 3) * 24),
            columns=list("ABC"),
        ),
    ]
)
def datetime_df(request: PytestFixtureRequest) -> pd.DataFrame:
    """DataFrame with datetime data, integer index, and str column names.
                   A          B          C
    0 2000-01-03 2000-01-02 2000-12-31
    1 2000-01-04 2000-01-09 2001-12-31
    2 2000-01-05 2000-01-16 2002-12-31
    3 2000-01-06 2000-01-23 2003-12-31
    4 2000-01-07 2000-01-30 2004-12-31
    5 2000-01-10 2000-02-06 2005-12-31
    6 2000-01-11 2000-02-13 2006-12-31
    7 2000-01-12 2000-02-20 2007-12-31
    8 2000-01-13 2000-02-27 2008-12-31
    9 2000-01-14 2000-03-05 2009-12-31"""
    df = pd.DataFrame(
        {
            "A": pd.date_range("2000-01-01", periods=10, freq="b"),
            "B": pd.date_range("2000-01-01", periods=10, freq="W"),
            "C": pd.date_range("2000-01-01", periods=10, freq="YE"),
        }
    )
    time_offset = request.param
    return df + time_offset


@pytest.fixture
def datetime_and_float_df(datetime_df: pd.DataFrame) -> pd.DataFrame:
    """datetime_df with additional columns of random positive and negative floats"""
    datetime_df[list("DE")] = np.random.uniform(-1, 1, (10, 2))
    return datetime_df


@pytest.fixture(autouse=True)
def _test_cache_cleanup() -> Iterator[None]:
    """
    Autouse fixture to remove EAB_tools/tests/io/data/.eab_tools_cache directory
    after each test.
    """
    # Before the test
    cache_path = Path(__file__).parent / "tests" / "io" / "data" / ".eab_tools_cache"
    cache_path.mkdir(exist_ok=True)
    pass

    # yield
    yield

    # After the test
    shutil.rmtree(cache_path)


@pytest.fixture(autouse=True)
def _docstring_tmp_path(request: PytestFixtureRequest) -> Iterator[None]:
    """
    Autouse  fixture to chdir to a tmp_path for all doctests without needing to
    explicitly call it.
    """
    # Almost completely adapted from a kind soul at https://stackoverflow.com/a/46991331
    # Trigger ONLY for the doctests.
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        # Get the fixture dynamically by its name.
        tmp_path: Path = request.getfixturevalue("tmp_path")
        # Chdir only for the duration of the test.
        og_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            yield
        finally:
            os.chdir(og_dir)
    else:
        # For normal tests, we have to yield, since this is a yield-fixture.
        yield
