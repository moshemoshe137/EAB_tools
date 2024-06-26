"""Tests for file I/O operations."""

import pytest

from EAB_tools import (
    sanitize_filename,
    sanitize_xl_sheetname,
)


class TestSanitizeFilename:
    """
    Contains tests for the `sanitize_filename` function.

    Methods
    -------
    test_sanitize_filename_very_simple:
        Check that valid filenames remain unchanged.
    test_sanitize_filename_simple:
        Verify that basic problematic characters are replaced.
    test_sanitize_filename_blank:
        Confirm that an empty string is returned as is.
    test_sanitize_filename_dirty_unicode:
        Special unicode characters should be stripped from filenames.
    """

    def test_sanitize_filename_very_simple(self) -> None:
        """
        Check that valid filenames remain unchanged.

        Ensure that filenames without any special characters are returned unchanged.
        """
        dirty_str = "EAB_tools are cool.csv"
        expected = dirty_str
        assert sanitize_filename(dirty_str) == expected

    def test_sanitize_filename_simple(self) -> None:
        """
        Verify that basic problematic characters are replaced.

        Ensure that common problematic characters in filenames, such as '?', are
        replaced with underscores.
        """
        dirty_str = "EAB_tools are cool?.csv"
        expected = "EAB_tools are cool_.csv"
        assert sanitize_filename(dirty_str) == expected

    def test_sanitize_filename_blank(self) -> None:
        """Confirm that an empty string is returned as is."""
        assert sanitize_filename("") == ""

    def test_sanitize_filename_dirty_unicode(self) -> None:
        """
        Special unicode characters should be stripped from filenames.

        Verifies that unicode and non-printable characters are effectively removed
        from filenames, ensuring file system compatibility.
        """
        dirty_str = "foo\0@🐍bar.png"
        assert sanitize_filename(dirty_str) == "foo___bar.png"


class TestSanitizeXlSheetname:
    r"""
    Contains tests for the `sanitize_xl_sheetname` function.

    Methods
    -------
    test_simple:
        Simple sheet names that comply with Excel limits should remain unchanged.
    test_too_long:
        Long sheet names should be truncated to meet Excel's character limit.
    test_blank:
        Verify that blank sheet names raise a ValueError.
    test_history:
        Ensure that 'history' is not allowed as a sheet name, in any case format.
    test_apostrophe_on_ends:
        Ensure apostrophes are not used at the start or end of sheet names.
    test_illegal_chars:
        Sheet names should not contain any illegal characters such as \/?*[]:"
    """

    def test_simple(self) -> None:
        """Simple sheet names that comply with Excel limits should remain unchanged."""
        dirty_str = "Sheet 1"
        expected = dirty_str
        assert sanitize_xl_sheetname(dirty_str) == expected

    def test_too_long(self) -> None:
        """
        Long sheet names should be truncated to meet Excel's character limit.

        Ensures that sheetnames longer than 31 characters are correctly truncated to
        meet Excel's maximum character limit.
        """
        dirty_str = (
            "This is my sheet name it's a very long sheet name "
            "I wish it could be shorter"
        )
        expected = "name I wish it could be shorter"
        sanitized = sanitize_xl_sheetname(dirty_str)
        assert expected == sanitized and len(sanitized) <= 31

    def test_blank(self) -> None:
        """Verify that blank sheet names raise a `ValueError`."""
        with pytest.raises(ValueError):
            sanitize_xl_sheetname("")

    @pytest.mark.parametrize("sn", ["History", "History", "HiStOrY"])
    def test_history(self, sn: str) -> None:
        """
        Ensure that "history" is not allowed as a sheetname in any case format.

        Parameters
        ----------
        sn : str
            The sheet name to test, intended to trigger a `ValueError`.
        """
        with pytest.raises(ValueError):
            sanitize_xl_sheetname(sn)

    strs = [
        "'This is bad'",
        "this's ok",
        "can't end'",
        "'can't start",
        "this is all good",
        "abba",
    ]

    @pytest.mark.parametrize("sn", strs)
    def test_apostrophe_on_ends(self, sn: str) -> None:
        """
        Ensure that apostrophes are not used at the start or end of sheet names.

        Apostrophes are allowed within sheet names, but not as the first or last
        character.

        Parameters
        ----------
        sn : str
            The sheet name to test for compliance with naming rules.
        """
        clean = sanitize_xl_sheetname(sn)
        assert clean[0] != "'" and clean[-1] != "'"

    strs = [
        r"No /slashes!",
        r"Of\any kind",
        r"C:\Windows\sys32",
        r"/bin/python3",
        r"No q marks?",
        r"No *****'n stars!",
        r"square =] brackets =[",
        "9:00 AM",
    ]

    @pytest.mark.parametrize("sn", strs)
    def test_illegal_chars(self, sn: str) -> None:
        r"""
        Sheet names should not contain any of these illegal characters.

        This test checks for the removal of these characters from sheet names. The
        following characters are illegal:
        ```
             \/?*[]:"
        ```

        Parameters
        ----------
        sn : str
            The sheet name to test for illegal characters.
        """
        illegal = list(r"/\?*[]:")
        assert all(bad_char not in sanitize_xl_sheetname(sn) for bad_char in illegal)
