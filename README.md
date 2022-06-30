# EAB Tools
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Tools for analyzing data exported from the EAB Navigate Student Success Management Software.

## Roadmap

This software is currently in prerelease. The plan is to start by packing up existing code from [my university](https://nl.edu/) (and possibly others). I plan to take a test-driven development philosophy.

## Installation

Use pip to install EAB_tools. You must have [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) or [Git for Windows](https://git-scm.com/download/win) installed.

```bash
pip install git+https://github.com/moshemoshe137/EAB_tools.git
```

To install in development mode, first create a virtual environment:
- On Unix/macOS run:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
- On Windows:
  - using Powershell:
      ```pwsh
        py -m venv .venv
        .venv\Scripts\Activate.ps1
      ```
  - using Command Prompt:
      ```cmd
        py -m venv .venv
        .venv\Scripts\activate.bat
      ```
Then install `EAB_tools` and its dependencies:
```bash
# Should work on Linux, macOS, and Windows:
git clone https://github.com/moshemoshe137/EAB_tools.git
cd EAB_tools
pip install --editable . --requirement ./dev-requirements.txt
# Pre-commit must be installed
python -m pre_commit install
# Test the installation
python -m pytest .
```

## Dependencies

Installing EAB tools will install these packages and their dependencies:

- [pandas](https://github.com/pandas-dev/pandas)
- [IPython](https://github.com/ipython/ipython)
- [dataframe_image](https://github.com/moshemoshe137/dataframe_image)

## License

[MIT](https://choosealicense.com/licenses/mit/)
