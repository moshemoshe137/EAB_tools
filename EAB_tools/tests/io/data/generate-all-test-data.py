"""Generate all data for tests from a single script."""

from pathlib import Path
import subprocess
import sys

parent_dir = Path(__file__).parent

# Generate test images
subprocess.run(
    [sys.executable, "-m", "IPython", f"{parent_dir / 'Generate Test Images.ipynb'}"]
)

# Generate fake Enrollment Reports
subprocess.run([sys.executable, f"{parent_dir / 'generate-fake-enrollments.py'}"])
for random_seed in [
    0,  # Zero
    28,  # Perfect number
    137,  # moshemoshe137
    1729,  # Taxicab number
    24601,  # Les Misérables
    525600,  # minutes
    1000000,  # dollars
    6.022e23,  # Avogadro's number
]:
    random_seed %= 2**32  # Maximum seed for `numpy`
    random_seed_str = str(int(random_seed))
    subprocess.run(
        [
            sys.executable,
            f"{parent_dir / 'generate-fake-enrollments.py'}",
            "--random-seed",
            random_seed_str,
        ]
    )
