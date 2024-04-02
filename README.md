# EAB Tools
[![License: MIT](https://img.shields.io/github/license/moshemoshe137/EAB_tools)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/EAB_tools)](https://pypi.org/project/EAB-tools/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![codecov](https://codecov.io/gh/moshemoshe137/EAB_tools/branch/dev/graph/badge.svg?token=IK9XOXSF3L)](https://codecov.io/gh/moshemoshe137/EAB_tools)

Tools for analyzing data exported from the EAB Navigate Student Success Management Software.

## Roadmap

This software is currently in prerelease. The plan is to start by packing up existing code from [my university](https://nl.edu/) (and possibly others). I plan to take a test-driven development philosophy.

## Installation

### Normal Usage:

Use pip to install EAB_tools:

```bash
pip install EAB_tools
```

### Development Mode:

To install in development mode, first create a virtual environment:
- On Unix/macOS run:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
- On Windows:
  - using PowerShell:
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
- [dataframe_image](https://github.com/dexplo/dataframe_image)

## Acknowledgements and Disclaimer
###  Acknowledgements
This project is not affiliated with, endorsed by, or sponsored by EAB Global, Inc.
"EAB Navigate" is a trademark of EAB Global, Inc. The use of the name "EAB Navigate"
in this project is for identification purposes only and does not imply any
association with EAB Global, Inc.

### Disclaimer

This independent project is developed for educational and informational purposes,
to assist users in analyzing data they have legally exported from the EAB Navigate
software. Users are responsible for ensuring their use of this tool complies with
EAB's Terms of Use and any other applicable EAB policies. This project does not
reproduce, distribute, or commercially exploit any EAB properties or services.

### Additional Notes

- This project's compatibility with EAB Navigate data relies on publicly available
data formats or legally exported data. It does not reverse engineer or derive
methods from EAB Navigate's proprietary technologies.

- For concerns about the use of the "EAB Navigate" trademark, [please contact
me](https://github.com/moshemoshe137).

- Efforts have been made to avoid confusion or misrepresentation of this project's
affiliation with EAB Global, Inc., or its products, including EAB Navigate.

- Mention of EAB Navigate within this project does not imply endorsement by
EAB Global, Inc.

## License

[MIT](https://choosealicense.com/licenses/mit/)
