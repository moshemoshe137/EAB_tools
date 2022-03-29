"""Tests for file io"""

from EAB_tools.io.filenames import sanitize_filename


class TestSanitizeFilename:
    def test_sanitize_filename_very_simple(self):
        """A (relatively) valid filename should remain untouched."""
        dirty_str = "EAB_tools are cool.csv"
        expected = dirty_str
        assert sanitize_filename(dirty_str) == expected

    def test_sanitize_filename_simple(self):
        """Basic troublesome chars should be replaced"""
        dirty_str = "EAB_tools are cool?.csv"
        expected = "EAB_tools are cool_.csv"
        assert sanitize_filename(dirty_str) == expected

    def test_sanitize_filename_blank(self):
        """A blank str should be untouched"""
        assert sanitize_filename('') == ''

    def test_sanitize_filename_dirty_unicode(self):
        """Special unicode characters should be removed"""
        dirty_str = "foo\0@🐍bar.png"
        assert sanitize_filename(dirty_str) == "foo___bar.png"
