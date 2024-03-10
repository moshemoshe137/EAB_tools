"""Functions used to generate fake data."""

from __future__ import annotations  # for Python 3.9

import random
from typing import Any

from faker import Faker
import numpy as np
import numpy.typing as npt
import pandas as pd


def make_probs_sum_to_one(d: dict[Any, float]) -> dict[Any, float]:
    """Make sure the probabilities in dict values add up to one."""
    return {key: value / sum(d.values()) for key, value in d.items()}


def EV(d: dict[float, float], normalize: bool = True) -> float:
    """Calculate Expected Value.

    Compute the expected value of a dict mapping `float`s to their probability or
    to their weight (the latter if `normalize=True`)
    """
    if normalize:
        d = make_probs_sum_to_one(d)
    return sum(key * value for key, value in d.items())


def sample_from_dict(
    distribution: dict[Any, float],
    size: int | None = None,
    normalize: bool = True,
    replace: bool = True,
) -> Any:
    """Take a sample from a `dict`.

    Take a sample from a `dict`, where the keys are the sample space and the
    values are the weights of each key
    """
    if normalize:
        distribution = make_probs_sum_to_one(distribution)
    keys, weights = zip(*distribution.items())
    return np.random.choice(keys, p=weights, size=size, replace=replace)


def generate_emails(
    names_series: pd.Series, existing_emails: pd.Series | None = None
) -> pd.Series:
    """Generate email addresses.

    Generate emails based on a list of `student_names` without duplicating any values in
    `existing_emails`
    """
    if existing_emails is None:
        existing_emails = pd.Series()

    # If `existing_emails` has duplicates, raise a `ValueError`
    if existing_emails.duplicated().any():
        duplicated_values = ", ".join(
            existing_emails[existing_emails.duplicated()].unique()
        )
        raise ValueError(
            f"`existing_emails` contains duplicate addresses: {duplicated_values}"
        )

    names = names_series.str.extract(r"^(?P<last>[\w '\-]+), (?P<first>[\w '\-]+)$")

    # Convert all names to ASCII equivalents
    # https://chat.openai.com/share/7db700d9-5647-44f4-927c-85b76efa1e9f
    first_names_ascii = (
        names["first"]
        .str.normalize("NFKD")
        .str.encode("ASCII", "ignore")
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.lower()
    )
    last_names_ascii = (
        names["last"]
        .str.normalize("NFKD")
        .str.encode("ASCII", "ignore")
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.lower()
    )

    base_emails = first_names_ascii + "." + last_names_ascii
    # Get the existing base-- that is, the part before "@example.edu"
    existing_base_emails = existing_emails.str.extract(
        r"(.+)@example\.edu", expand=False
    )

    # Concatenate and reset index to align with base_emails
    all_emails = pd.concat([existing_base_emails, base_emails], ignore_index=True)
    # Remove periods for comparison
    all_emails_normalized = all_emails.str.replace(".", "").str.lower()
    # Find duplicates in normalized emails
    duplicates = all_emails_normalized.duplicated(keep="first")

    number = 2  # Start numbering at 2
    while duplicates.any():
        # Extract the beginning part of the email base, up to (but ignoring) the number
        all_emails.loc[duplicates] = all_emails[duplicates].str.extract(
            r"^([A-Za-z\.]+)\d?", expand=False
        ) + str(number)
        all_emails_normalized = all_emails.str.replace(".", "").str.lower()
        duplicates = all_emails_normalized.duplicated(keep="first")
        number += 1

    # Be sure to only return the part of `all_emails` that's AFTER the `existing_emails`
    new_emails = all_emails[len(existing_emails) :] + "@example.edu"
    return new_emails.reset_index(drop=True)


def select_categories(length: int = 1) -> npt.NDArray[np.str_]:
    """Generate students' categories."""
    # Create a DataFrame to hold each category dimension
    categories = pd.DataFrame(index=range(length))

    # Define some probabilities
    campus_options = {"On-line": 0.5, "Main Campus": 0.4, "Satellite Campus": 0.1}
    graduated_options = {  # Mutually exclusive with certain other categories
        "Graduated: Yes": 0.1,
        "Graduated: No": 0.9,
    }
    level_options = [
        "Undergraduate",
        "Graduate",
    ]
    hold_types = ["Hold: Financial", "Hold: Academic", "Hold: Administrative"]
    comp_rate_options = {
        "Completion Rate: >= 66.67%": 0.75,
        "Completion Rate: < 66.67%": 0.25,
    }
    start_term_options = [
        f"Start Term: {season} {year}"
        for year in range(2000, 2025)
        for season in ["Fall", "Winter", "Spring", "Summer"]
    ]
    start_term_options_dict = {
        term: float(i) for i, term in enumerate(start_term_options, start=1)
    }
    term_status_options = {  # Mutually exclusive with "Graduated: Yes"
        "Term Status: Registered": 0.85,
        "Term Status: Not Registered": 0.15,
    }
    fafsa_options = [  # Mutually exclusive with "Graduated: Yes"
        "FAFSA: Yes",
        "FAFSA: No",
    ]

    # Helper func
    def _include_category_mask(prob: float = 0.9) -> npt.NDArray[np.bool_]:
        return np.random.rand(length) < prob

    # Campus
    include_campus_mask = _include_category_mask()
    categories.loc[include_campus_mask, "campus"] = sample_from_dict(
        campus_options, sum(include_campus_mask)
    )

    # Graduated
    include_graduated_mask = _include_category_mask()
    categories.loc[include_graduated_mask, "graduated"] = sample_from_dict(
        graduated_options, sum(include_graduated_mask)
    )

    # Level
    include_level_mask = _include_category_mask()
    categories.loc[include_level_mask, "level"] = np.random.choice(
        level_options, sum(include_level_mask)
    )

    # Hold
    # Only 10% chance of showing a Hold
    include_hold_mask = _include_category_mask(prob=0.10)
    categories.loc[include_hold_mask, "hold"] = np.random.choice(
        hold_types, sum(include_hold_mask)
    )

    # Completion rate
    include_comp_rate_mask = _include_category_mask()
    categories.loc[include_comp_rate_mask, "comp_rate"] = sample_from_dict(
        comp_rate_options, sum(include_comp_rate_mask)
    )

    # Start Term
    include_start_term_mask = _include_category_mask()
    categories.loc[include_start_term_mask, "start_term"] = sample_from_dict(
        start_term_options_dict, sum(include_start_term_mask)
    )

    # Term Status (Registered/Not Registered)
    # Mutually exclusive with "Graduated: Yes"
    not_graduated_mask = ~(categories["graduated"] == "Graduated: Yes")
    include_status_mask = _include_category_mask() & not_graduated_mask
    categories.loc[include_status_mask, "status"] = sample_from_dict(
        term_status_options, sum(include_status_mask)
    )

    # FAFSA
    # Mutually exclusive with "Graduated: Yes"

    include_fafsa_mask = _include_category_mask() & not_graduated_mask
    categories.loc[include_fafsa_mask, "FAFSA"] = np.random.choice(
        fafsa_options, sum(include_fafsa_mask)
    )

    # Join the categories and return the NumPy array
    categories_strs = categories.apply(
        lambda row: ", ".join(row.dropna()), axis="columns"
    ).to_numpy(dtype=np.str_)

    return categories_strs


def select_tag() -> str | None:
    """Generate a single student's tags."""
    tags = []

    # General tags
    general_tags = [
        "Honor Student",
        "Scholarship Recipient",
        "At Risk",
        "Needs Tutoring",
        "Athlete",
        "International",
        "Transfer",
    ]
    # Tuples of mutually exclusive tags
    mutually_exclusive_tags = [("Part-Time", "Full-Time")]

    max_tags = len(general_tags) + len(mutually_exclusive_tags)

    def _include_tag(chance: float = 1 - 0.675 ** (1 / max_tags)) -> bool:
        """Randomly decide to include a tag or not.

        We want a 67.5% chance that a student has no tags whatsoever.
        With 8 possible tags, a little algebra reveals that the default chance must be
        `1 - 0.675 ** (1 / num_possible_tags)`.
        """
        return np.random.rand() < chance

    for tag in general_tags:
        if _include_tag():
            tags.append(tag)

    for tuple_of_mutually_exclusive_tags in mutually_exclusive_tags:
        if _include_tag():
            tags.append(np.random.choice(tuple_of_mutually_exclusive_tags))

    return ", ".join(tags) if tags else None


def select_tags(length: int = 1) -> npt.NDArray[np.str_]:
    """Generate students' tags."""
    return np.array([select_tag() for _ in range(length)])


def generate_cumulative_gpas(
    length: int = 1,
) -> npt.NDArray[np.floating[npt.NBitBase]]:
    """Generate students' cumulative GPAs."""
    # Probabilities
    prob_of_zero_gpa = 0.12
    prob_of_high_gpa = 0.75 * (1 - prob_of_zero_gpa)  # 75% of non-zero

    # Distribution counts
    zero_gpas_count = np.random.binomial(length, prob_of_zero_gpa)
    high_gpas_count = np.random.binomial(length - zero_gpas_count, prob_of_high_gpa)
    low_gpas_count = length - zero_gpas_count - high_gpas_count

    # Generate GPAs
    zero_gpas = np.zeros(zero_gpas_count)
    high_gpas = np.clip(np.random.normal(3.9, 0.4, size=high_gpas_count), 0, 4)
    low_gpas = np.random.uniform(low=0, high=2.6, size=low_gpas_count)

    # Combine and shuffle
    all_gpas = np.concatenate([zero_gpas, high_gpas, low_gpas]).round(2)
    np.random.shuffle(all_gpas)

    all_gpas

    return np.round(all_gpas, 2)


def generate_staff_df(
    fake: Faker,
    assigned_staff_role_probabilities: dict[str, float],
    n_staff: int = 1,
    existing_emails: pd.Series | None = None,
    existing_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate a table of staff info for `select_assigned_staff`."""
    staff_df = pd.DataFrame()

    locales_dict = dict(zip(fake.locales, fake.weights if fake.weights else [1]))

    staff_df["name"] = [
        f"{fake[locale].last_name()}, {fake[locale].first_name()}"
        for locale in sample_from_dict(locales_dict, n_staff)
    ]

    staff_df["email"] = generate_emails(
        staff_df["name"], existing_emails=existing_emails
    )

    staff_df["id"] = pd.NA
    existing_ids = pd.Series() if existing_ids is None else existing_ids
    while staff_df["id"].isna().any():
        # Repeat the ID number assignment if anyone overlaps with a Student ID (even tho
        # this is EXTREMELY unlikely!)
        staff_df["id"] = [
            (f"ID{rand_id:09}" if f"ID{rand_id:09}" not in existing_ids else pd.NA)
            for rand_id in random.sample(range(10**9), n_staff)
        ]

    staff_df["role"] = sample_from_dict(assigned_staff_role_probabilities, n_staff)

    return staff_df


def select_assigned_staff(
    staff_df: pd.DataFrame,
    probabilities_dict: dict[str, float],
    length: int = 1,
    include_professor: bool = False,
) -> npt.NDArray[np.str_]:
    if not include_professor:
        probabilities_dict = {
            role: prob
            for role, prob in probabilities_dict.items()
            if role.casefold() != "professor"
        }

    assigned_staff = pd.Series(dtype="string", index=range(length))
    for staff_role, prob in probabilities_dict.items():
        this_role_df = staff_df[staff_df["role"] == staff_role]
        mask_has_this_staff = np.random.rand(length) < prob
        n_has_this_staff = mask_has_this_staff.sum()

        chosen_staff = pd.Series(
            np.random.choice(this_role_df["name"], n_has_this_staff)
        )

        staff_name_formatted = chosen_staff.str.replace(
            r"^(?P<last>[\w '\-]+), (?P<first>[\w '\-]+)$",
            rf"\g<first> \g<last> ({staff_role})",
            regex=True,
        )

        mask_this_is_my_first_staff = assigned_staff.isna() & mask_has_this_staff
        n_this_is_my_first_staff = mask_this_is_my_first_staff.sum()
        assigned_staff.loc[mask_this_is_my_first_staff & mask_has_this_staff] = (
            staff_name_formatted[:n_this_is_my_first_staff].values
        )
        assigned_staff.loc[~mask_this_is_my_first_staff & mask_has_this_staff] += (
            ", " + staff_name_formatted[n_this_is_my_first_staff:].values
        )

    return assigned_staff.values


__all__ = [
    "make_probs_sum_to_one",
    "EV",
    "sample_from_dict",
    "generate_emails",
    "select_categories",
    "select_tag",
    "select_tags",
    "generate_cumulative_gpas",
    "generate_staff_df",
]
