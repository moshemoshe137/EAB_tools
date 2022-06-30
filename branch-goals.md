
# Goals for this branch

- [Reddit thread](https://www.reddit.com/r/Python/comments/vb3tw6/comment/ic6ql46/?utm_source=share&utm_medium=web2x&context=3) mentioned some great linters:
    - ~~black~~
    - isort
    - ~~flake8~~
    - pylint
    - mypy
    - ~~pre-commit~~
- `pandas` also uses some linters:
    - ~~black~~
    - ~~flake8~~
    - ~~flake8-bugbear~~  `# used by flake8, find likely bugs`
    - ~~flake8-comprehensions~~  `# used by flake8, linting of unnecessary comprehensions`
    - isort  `# check that imports are in the right order`
    - mypy
    - ~~pre-commit~~
    - ~~pycodestyle~~  `# used by flake 8`
    - pyupgrade


Notably, pylint seems a little controversial and difficult to maintain. Still, doesn't necessarily hurt to have. When in doubt, I think I plan to follow [pandas](https://github.com/pandas-dev/pandas)'s conventions.
