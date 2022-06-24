# EAB Tools
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tools for analyzing data exported from the EAB Navigate Student Success Management Software.

## Roadmap

This software is currently in prerelease. The plan is to start by packing up existing code from [my university](https://nl.edu/) (and possibly others). I plan to take a test-driven development philosophy.

## Installation

Use pip to install EAB_tools. You must have [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) or [Git for Windows](https://git-scm.com/download/win) installed.

```bash
pip install git+https://github.com/moshemoshe137/EAB_tools.git
```

or in Development mode:

```bash
git clone https://github.com/moshemoshe137/EAB_tools.git
pip install --editable .
```

## Dependencies

Installing EAB tools will install these packages and their dependencies:

- [pandas](https://github.com/pandas-dev/pandas)
- [IPython](https://github.com/ipython/ipython)
- [dataframe_image](https://github.com/moshemoshe137/dataframe_image)

## License

[MIT](https://choosealicense.com/licenses/mit/)
