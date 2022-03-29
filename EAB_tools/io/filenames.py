"""Utilities for dealing with filenames and Excel sheet names."""

import re


def sanitize_filename(filename: str) -> str:
    """
    Ensure valid filenames.

    Given a filename, remove all characters that are
    potentially hazardous in a filename.
    The only chars allowed are

    - Word characters ([a-zA-Z0-9_])
    - Dashes
    - Periods
    - Spaces
    - Parenthesis

    Parameters
    ----------
    filename : str
        The filename to clean.

    Returns
    -------
    str
        The cleaned up filename.

    Examples
    --------
    >>> sanitize_filename('python is fun 🐍.py')
    'python is fun _.py'
    """
    return re.sub(r'[^\w\-_. ()]', '_', filename)
