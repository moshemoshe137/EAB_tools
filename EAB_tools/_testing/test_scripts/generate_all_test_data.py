"""Generate all data for tests from a single script."""

from pathlib import Path
import subprocess
import sys

import numpy as np

from EAB_tools._testing.test_scripts.generate_fake_enrollments import (
    generate_fake_enrollments,
)

RANDOM_SEEDS = [
    0,  # Zero
    28,  # Perfect number
    137,  # moshemoshe137
    1729,  # Taxicab number
    24601,  # Les Misérables
    525600,  # minutes
    1000000,  # dollars
    6.022e23,  # Avogadro's number
]

parent_dir = Path(__file__).parent


def main(
    generate_images: bool = True,
    generate_csvs: bool = True,
    generate_enrollment_reports: bool = True,
    RANDOM_SEEDS: list[float] = RANDOM_SEEDS,
) -> None:
    """Generate all data for tests from a single script."""
    if generate_images:
        # Generate test images
        subprocess.run(
            [
                sys.executable,
                "-m",
                "IPython",
                f"{parent_dir/'generate_test_images.ipynb'}",
            ]
        )

    if generate_csvs and generate_enrollment_reports:
        # Generate fake Enrollment Reports
        generate_fake_enrollments()
        for random_seed in RANDOM_SEEDS:
            random_seed %= 2**32  # Maximum seed for `numpy`
            generate_fake_enrollments(
                int(random_seed),
                int(np.clip(random_seed, 1, 10**6)) if random_seed > 0 else None,
            )


if __name__ == "__main__":
    main()
