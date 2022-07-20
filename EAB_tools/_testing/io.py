import tkinter
from types import TracebackType

from EAB_tools.io.io import PathLike


def _is_tkinter_error(
    err: tuple[type, Exception, TracebackType],
    *args: object,  # Flaky will pass more objects that I don't care about
) -> bool:
    return isinstance(err[1], tkinter.TclError)


def _test_photos_are_equal(base: PathLike, other: PathLike) -> bool:
    # https://stackoverflow.com/a/34669225
    return open(base, "rb").read() == open(other, "rb").read()
