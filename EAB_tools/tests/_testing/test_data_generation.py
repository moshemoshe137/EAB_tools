from __future__ import annotations  # for Python 3.9

from pathlib import Path
import random
import subprocess
import sys

from faker import Faker
import numpy as np
import pandas as pd
import pytest

from EAB_tools._testing.data_generation import (
    EV,
    generate_cumulative_gpas,
    generate_emails,
    generate_staff_df,
    make_probs_sum_to_one,
    sample_from_dict,
    select_assigned_staff,
    select_categories,
    select_tags,
)
from EAB_tools._testing.test_scripts.generate_fake_enrollments import (
    generate_fake_enrollments,
)

seeds = [0, 1, 987_654_321]


@pytest.fixture(params=seeds, ids=list(map("seed={:_}".format, seeds)))
def set_random_seed(request: pytest.FixtureRequest) -> int:
    random.seed(request.param, version=2)
    np.random.seed(request.param)
    Faker.seed(request.param)
    return request.param


class TestDataGeneration:
    def test_make_probs_sum_to_one(self) -> None:
        d1 = {"heads": 0.5, "tails": 0.5}
        assert d1 == make_probs_sum_to_one(d1)

        d2 = {"heads": 50.0, "tails": 50.0}
        assert make_probs_sum_to_one(d2) == d1

        d3 = {0: 17.0, 1: 13.0, 2: 10.0, 3: 10.0}
        d3_expected = {0: 17 / 50, 1: 13 / 50, 2: 1 / 5, 3: 1 / 5}
        assert make_probs_sum_to_one(d3) == pytest.approx(d3_expected)

    def test_sample_from_dict(self) -> None:
        weights_dict = {"heads": 49.5, "tails": 49.5, "you broke the coin": 1}
        sample = sample_from_dict(weights_dict, 10_000)
        assert "you broke the coin" in sample

        assert sample.shape == (10_000,)

    def test_EV(self) -> None:
        assert EV({10: 1}) == 10
        assert EV({1: 1 / 6, 2: 2 / 3, 3: 1 / 6}) == 2

    def test_EV_all_zero_probs(self) -> None:
        with pytest.raises(ZeroDivisionError):
            EV({1: 0})

    def test_generate_cumulative_GPAs_expected_values(
        self, set_random_seed: int
    ) -> None:
        expected = {
            0: np.array([1.57]),
            1: np.array([0.00]),
            987_654_321: np.array([4.00]),
        }

        assert generate_cumulative_gpas() == expected[set_random_seed]

    def test_generate_cumulative_GPAs_expected_attributes(self) -> None:
        gpas = generate_cumulative_gpas(length=1_000)

        # Ensure the right shape
        assert gpas.shape == (1000,)

        # Ensure 0.00 <= GPA <= 4.00
        assert (gpas >= 0).all() and (gpas <= 4).all()

        assert gpas.mean() == pytest.approx(3.00, abs=0.5)

        assert gpas.std() == pytest.approx(1.50, abs=0.1)

    names = pd.Series(
        [
            "Doe, John",
            "Doe, John",
            "Doe, Joshe",
            "Dee, John",
            "Oe, Johnd",
            "Álvarez, José",
            "Blanchard, Zoë",
            "Dubois, François",
            "Müller, Jürgen",
            "Nilsen, Øystein",
            "Öztürk, İbrahim",
            "Bērziņš, Pēteris",
            "De Lucci, Chiara",
            "Larsen, Søren",
            "Żyła, Ewa",
        ]
    )

    def test_generate_expected_emails(self) -> None:

        emails = generate_emails(self.names)
        expected_emails = [
            "john.doe@example.edu",
            "john.doe2@example.edu",
            "joshe.doe@example.edu",
            "john.dee@example.edu",
            "johnd.oe3@example.edu",
            "jose.alvarez@example.edu",
            "zoe.blanchard@example.edu",
            "francois.dubois@example.edu",
            "jurgen.muller@example.edu",
            "ystein.nilsen@example.edu",
            "ibrahim.ozturk@example.edu",
            "peteris.berzins@example.edu",
            "chiara.delucci@example.edu",
            "sren.larsen@example.edu",
            "ewa.zya@example.edu",
        ]
        assert (emails == expected_emails).all()

        emails2 = generate_emails(self.names, existing_emails=emails)
        expected_emails = [
            "john.doe4@example.edu",
            "john.doe5@example.edu",
            "joshe.doe2@example.edu",
            "john.dee2@example.edu",
            "johnd.oe6@example.edu",
            "jose.alvarez2@example.edu",
            "zoe.blanchard2@example.edu",
            "francois.dubois2@example.edu",
            "jurgen.muller2@example.edu",
            "ystein.nilsen2@example.edu",
            "ibrahim.ozturk2@example.edu",
            "peteris.berzins2@example.edu",
            "chiara.delucci2@example.edu",
            "sren.larsen2@example.edu",
            "ewa.zya2@example.edu",
        ]
        assert (emails2 == expected_emails).all()

        assert not pd.concat([emails, emails2]).duplicated().any()

    def test_duplicated_existing_emails(self) -> None:
        with pytest.raises(ValueError, match="duplicate addresses:"):
            generate_emails(
                self.names, existing_emails=pd.Series(["foo.bar@example.edu"] * 2)
            )

    def test_select_categories_expected_values(self, set_random_seed: int) -> None:
        expected_values = {
            0: (
                "Main Campus, Graduated: No, Graduate, Completion Rate: >= 66.67%,"
                " Start Term: Fall 2017, Term Status: Registered, FAFSA: No"
            ),
            1: (
                "Main Campus, Graduated: No, Undergraduate, Completion Rate: >= 66.67%,"
                " Term Status: Registered, FAFSA: Yes"
            ),
            987_654_321: "On-line, Graduated: No, Graduate, FAFSA: Yes",
        }
        assert select_categories()[0] == expected_values[set_random_seed]

    def test_select_tags_expected_values(self, set_random_seed: int) -> None:
        expected_values = {
            0: [None, None, "Honor Student", None],
            1: ["At Risk", "Transfer", None, "Needs Tutoring"],
            987_654_321: [None, "Athlete", None, None],
        }
        tags = select_tags(4)
        assert (tags == expected_values[set_random_seed]).all()

    def test_selected_tags_mutually_exclusive(self) -> None:
        tags = pd.Series(select_tags(100_000))
        mutually_exclusive_tags = [("Part-Time", "Full-Time")]
        for mutually_exclusive_tag_set in mutually_exclusive_tags:
            mask = pd.Series(True)
            for mutually_exclusive_tag in mutually_exclusive_tag_set:
                mask &= tags.str.contains(mutually_exclusive_tag)
        assert mask.sum() == 0

    assigned_staff_role_probabilities: dict[str, float] = {
        "Advisor": 0.90,
        "Career Advisor": 0.80,
        "Professor": 0.20,
        "Student Finance": 0.10,
        "FAFSA coordinator": 0.10,
        "International Success Advisor": 0.02,
        "Field Advisor": 0.01,
        "VA coordinator": 0.01,
    }

    faker_locales_params = [
        None,
        {"en_US": 90, "es_MX": 5, "en_CA": 2, "en_GB": 1, "fr_FR": 1, "de_DE": 1},
    ]

    @pytest.fixture(
        params=faker_locales_params, ids=list(map(str, faker_locales_params))
    )
    def faker_locales(self, request: pytest.FixtureRequest) -> dict[str, float] | None:
        return request.param

    @pytest.fixture
    def staff_df(
        self, set_random_seed: int, faker_locales: dict[str, float] | None
    ) -> pd.DataFrame:
        return generate_staff_df(
            Faker(faker_locales), self.assigned_staff_role_probabilities, n_staff=1_000
        )

    def test_assigned_staff_works(
        self,
        set_random_seed: int,
        staff_df: pd.DataFrame,
        faker_locales: dict[str, float] | None,
    ) -> None:

        expected = (
            {
                0: "Michelle Kaiser (Advisor), Luke Miller (Career Advisor)",
                1: "Morgan Bradley (Advisor), Tiffany Nielsen (Career Advisor)",
                987_654_321: "Erin Richard (Advisor), Monica Mitchell (Career Advisor)",
            }
            if faker_locales is None
            else {
                0: "Stacey Simpson (Advisor), Christopher Daniels (Career Advisor)",
                1: "Mary Andrews (Advisor), Sharon Gonzalez (Career Advisor)",
                987_654_321: (
                    "Cindy Mccormick (Advisor), Jessica Hughes (Career Advisor)"
                ),
            }
        )
        staff = select_assigned_staff(staff_df, self.assigned_staff_role_probabilities)
        assert staff[0] == expected[set_random_seed]


class TestReportGeneration:
    def test_generate_fake_enrollments(self, set_random_seed: int) -> None:
        # Base `n_records` on the seed
        n_records = set_random_seed + 1000
        n_records %= 10**5  # But cap it at 10,000 records

        generate_fake_enrollments(RANDOM_SEED=set_random_seed, n_records=n_records)

    @pytest.mark.parametrize(
        "random_seed_flag", ["-r", "--random-seed"], ids=" {} ".format
    )
    @pytest.mark.parametrize("n_records_flag", ["-n", "--n-records"], ids=" {} ".format)
    def test_generate_fake_enrollments_cli(
        self, set_random_seed: int, random_seed_flag: str, n_records_flag: str
    ) -> None:
        # Base `n_records` on the seed
        n_records = set_random_seed + 1000
        n_records %= 10**5  # But cap it at 10,000 records

        script_location = (
            Path(__file__)
            / "../../.."
            / "_testing"
            / "test_scripts"
            / "generate_fake_enrollments.py"
        )
        subprocess.run(
            [
                sys.executable,
                # r"EAB_tools\tests\_testing\test_data_generation.py",
                script_location,
                random_seed_flag,
                str(set_random_seed),
                n_records_flag,
                str(n_records),
            ],
            check=True,
        )
