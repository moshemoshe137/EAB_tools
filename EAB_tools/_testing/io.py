import tkinter
from types import TracebackType

from matplotlib.backends.backend_tkagg import FigureManagerTk
import matplotlib.pyplot as plt

from EAB_tools._testing.types import PathLike


def _is_tkinter_error(
    err: tuple[type, Exception, TracebackType],
    *args: object,  # Flaky will pass more objects that I don't care about
) -> bool:
    return isinstance(err[1], tkinter.TclError)


def _test_photos_are_equal(base: PathLike, other: PathLike) -> bool:
    # https://stackoverflow.com/a/34669225
    return open(base, "rb").read() == open(other, "rb").read()


def _minimize_tkagg() -> None:
    if plt.get_backend().casefold() == "tkagg":
        # Rapidly minimizes the window to prevent strobing effect.
        # Works on my Windows 10, at least...
        fig_manager = plt.get_current_fig_manager()
        assert isinstance(fig_manager, FigureManagerTk)
        fig_manager.window.state("iconic")
