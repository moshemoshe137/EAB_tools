# %%
# This code was mostly created by ChatGPT
"""Generate a fake EAB v2 Enrollments Report."""
from __future__ import annotations

from pathlib import Path
import random

from faker import Faker
import numpy as np
import pandas as pd

import EAB_tools
from EAB_tools._testing.data_generation import (
    EV,
    generate_cumulative_gpas,
    generate_emails,
    generate_staff_df,
    sample_from_dict,
    select_assigned_staff,
    select_categories,
    select_tags,
)

# %%
locales_dict: dict[str, float] = {
    "en_US": 90,
    "es_MX": 5,
    "en_CA": 2,
    "en_GB": 1,
    "fr_FR": 1,
    "de_DE": 1,
}
fake = Faker(locales_dict)
np.random.seed(42)  # Also sets the random seed for `pandas`
Faker.seed(42)
random.seed(42)

# %%
n_records = 85_253

# %%
sections_per_course_distribution: dict[float, float] = {
    1: 0.5288,
    2: 0.2028,
    3: 0.0946,
    4: 0.0601,
    5: 0.0274,
    6: 0.0286,
    7: 0.0211,
    8: 0.0033,
    9: 0.0030,
    10: 0.0051,
    11: 0.0079,
    12: 0.0031,
    13: 0.0032,
    15: 0.0029,
    18: 0.0014,
    19: 0.0016,
    20: 0.0026,
    24: 0.0012,
    35: 0.0013,
}

instructors_per_course_distribution: dict[float, float] = {
    0: 0.0609,
    1: 0.9224,
    2: 0.0168,
}

sections_per_instructor_distribution: dict[float, float] = {
    1: 0.3148,
    2: 0.3148,
    3: 0.1872,
    4: 0.0922,
    5: 0.0425,
    6: 0.0192,
    7: 0.0149,
    8: 0.0055,
    9: 0.0036,
    10: 0.0035,
    15: 0.0018,
}

students_per_section_distribution: dict[float, float] = {
    1: 0.0579,
    2: 0.0373,
    3: 0.0290,
    4: 0.0413,
    5: 0.0284,
    6: 0.0472,
    7: 0.0332,
    8: 0.0263,
    9: 0.0402,
    10: 0.0409,
    11: 0.0329,
    12: 0.0358,
    13: 0.0276,
    14: 0.0274,
    15: 0.0225,
    16: 0.0326,
    17: 0.0298,
    18: 0.0215,
    19: 0.0219,
    20: 0.0158,
    21: 0.0175,
    22: 0.0194,
    23: 0.0137,
    24: 0.0176,
    25: 0.0211,
    26: 0.0209,
    27: 0.0094,
    28: 0.0167,
    29: 0.0246,
    30: 0.0110,
    31: 0.0151,
    32: 0.0112,
    33: 0.0122,
    34: 0.0130,
    35: 0.0148,
    36: 0.0157,
    37: 0.0073,
    38: 0.0159,
    39: 0.0080,
    40: 0.0091,
    41: 0.0053,
    42: 0.0057,
    43: 0.0065,
    44: 0.0040,
    45: 0.0016,
    46: 0.0033,
    47: 0.0046,
    48: 0.0027,
    49: 0.0015,
    50: 0.0040,
    51: 0.0035,
    52: 0.0016,
    53: 0.0012,
    54: 0.0019,
    55: 0.0010,
    56: 0.0020,
    57: 0.0020,
    59: 0.0013,
    60: 0.0006,
    63: 0.0005,
    69: 0.0012,
    111: 0.0005,
}

classifications_distribution: dict[str, float] = {
    "Foo (Winter 2024)": 0.6120,
    "Foo (Fall 2023)": 0.1859,
    "Graduate (Winter 2024)": 0.0573,
    "None": 0.0379,
    "Foo (Spring 2024)": 0.0369,
    "Graduate (Fall 2023)": 0.0160,
    "Foo (Spring 2023)": 0.0151,
    "Foo (Summer 2023)": 0.0130,
    "Foo (Winter 2023)": 0.0064,
    "Foo (Fall 2022)": 0.0053,
    "Graduate (Spring 2023)": 0.0015,
    "Foo (Spring 2022)": 0.0017,
    "Graduate (Summer 2023)": 0.0014,
    "Graduate (Spring 2024)": 0.0013,
    "Graduate (Winter 2023)": 0.0010,
    "Graduate (Fall 2022)": 0.0005,
    "Graduate (Summer 2022)": 0.0005,
    "Foo (Fall 2021)": 0.0005,
    "Foo (Spring 2020)": 0.0004,
    "Foo (Fall 2019)": 0.0003,
    "Foo (Summer 2024)": 0.0004,
    "Foo (Winter 2022)": 0.0004,
    "Foo (Summer 2022)": 0.0003,
    "Graduate (Fall 2018)": 0.0002,
    "Graduate (Winter 2022)": 0.0002,
    "Foo (Fall 2020)": 0.0002,
    "Foo (Winter 2019)": 0.0002,
    "Foo (Winter 2020)": 0.0002,
    "Graduate (Winter 2018)": 0.0002,
    "Graduate (Spring 2021)": 0.0002,
    "Foo (Winter 2017)": 0.0002,
    "Graduate (Winter 2021)": 0.0002,
    "Graduate (Summer 2021)": 0.0002,
    "Freshman (Winter 2024)": 0.0001,
    "Foo (Winter 2021)": 0.0001,
    "Foo (Spring 2016)": 0.0001,
    "Graduate (Spring 2017)": 0.0001,
    "Graduate (Summer 2020)": 0.0001,
    "Graduate (Spring 2020)": 0.0001,
    "Foo (Spring 2019)": 0.0001,
    "Graduate (Fall 2020)": 0.0001,
    "Graduate (Spring 2019)": 0.0001,
    "Graduate (Spring 2022)": 0.0001,
    "Foo (Summer 2021)": 0.0001,
    "Graduate (Fall 2021)": 0.0001,
    "Foo (Fall 2016)": 0.0001,
    "Foo (Summer 2019)": 0.0001,
    "Freshman (Fall 2022)": 0.0001,
    "Graduate (Summer 2024)": 0.0001,
    "Foo (Fall 2024)": 0.0001,
    "Foo (Spring 2021)": 0.0001,
}

num_courses_per_student_distribution: dict[float, float] = {
    1: 0.1870,
    2: 0.2159,
    3: 0.2484,
    4: 0.1562,
    5: 0.0881,
    6: 0.0480,
    7: 0.0226,
    8: 0.0105,
    9: 0.0067,
    10: 0.0056,
    11: 0.0018,
    12: 0.0053,
    13: 0.0027,
    14: 0.0005,
    15: 0.0001,
    16: 0.0005,
    17: 0.0002,
}

credit_hours_per_student_distribution: dict[float, float] = {
    0: 0.14,
    1: 0.05,
    2: 0.07,
    3: 0.26,
    4: 0.01,
    5: 0.44,
    6: 0.02,
    12: 0.01,
}

start_times_distribution: dict[str, float] = {
    "7:00 AM": 0.01,
    "8:00 AM": 0.06,
    "9:00 AM": 0.29,
    "10:00 AM": 0.06,
    "10:30 AM": 0.03,
    "11:00 AM": 0.06,
    "12:00 PM": 0.01,
    "1:00 PM": 0.29,
    "2:00 PM": 0.03,
    "3:00 PM": 0.05,
    "4:00 PM": 0.05,
    "4:30 PM": 0.05,
    "5:00 PM": 0.02,
    "6:00 PM": 0.01,
    "6:30 PM": 0.01,
}

course_duration_distribution: dict[pd.Timedelta, float] = {
    pd.to_timedelta(time): prob
    for time, prob in {
        "30 min": 0.01,
        "1 hr": 0.03,
        "75 min": 0.17,
        "2 hr": 0.05,
        "150 min": 0.05,
        "3 hr": 0.56,
        "4 hr": 0.05,
        "5 hr": 0.04,
        "6 hr": 0.02,
        "7 hr": 0.01,
        "8 hr": 0.01,
    }.items()
}

class_days_distribution: dict[str | None, float] = {
    None: 0.14,
    "MWF": 0.14,
    "TR": 0.14,
    "T": 0.11,
    "R": 0.10,
    "MW": 0.10,
    "W": 0.08,
    "M": 0.07,
    "MTWRF": 0.05,
    "Sa": 0.03,
    "F": 0.03,
    "WR": 0.01,
    "MTWR": 0.01,
}

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

# %%
avg_sections_per_course = EV(sections_per_course_distribution)
avg_instructors_per_course = EV(instructors_per_course_distribution)
avg_sections_per_instructor = EV(sections_per_instructor_distribution)
avg_students_per_section = EV(students_per_section_distribution)
avg_courses_per_student = EV(num_courses_per_student_distribution)
avg_credit_hours_per_student = EV(credit_hours_per_student_distribution)

# %%
# Calculate the number of unique students based on the average number of courses per
# student
n_unique_students = int(n_records / avg_courses_per_student)

majors = (
    "Accounting,Anthropology,Biochemistry,Biological Sciences,Business Administration,"
    "Chemical Engineering,Civil Engineering,Computer Science,Economics,"
    "Electrical Engineering,English Literature,Environmental Science,Finance,"
    "Graphic Design,History,Information Technology,Journalism,Marketing,Mathematics,"
    "Mechanical Engineering,Music,Nursing,Philosophy,Physics,Political Science,"
    "Psychology,Sociology,Software Engineering,Statistics,Theater Arts"
).split(",")

# Generate unique student data
unique_students = pd.DataFrame(
    {
        "Student ID": [
            # The builtin `random.sample` method from the Python standard lib can
            # efficiently sample from `range` objects, while `np.random.sample` needs to
            # construct the entire list in memory first.
            f"ID{rand_id:09}"
            for rand_id in random.sample(range(10**9), n_unique_students)
        ],
        "Student Alternate ID": np.NaN,
        "Student Name": [
            f"{fake[locale].last_name()}, {fake[locale].first_name()}"
            for locale in sample_from_dict(locales_dict, size=n_unique_students)
        ],
        "Classification": sample_from_dict(
            classifications_distribution, size=n_unique_students
        ),
        "Major": np.random.choice(majors, n_unique_students),
        "Credit Hours": sample_from_dict(
            credit_hours_per_student_distribution, size=n_unique_students
        ),
    }
).replace("None", np.NaN)

# %%
unique_students["Student E-mail"] = generate_emails(unique_students["Student Name"])

# %%
existing_emails = unique_students["Student E-mail"].copy()
existing_ids = unique_students["Student ID"].copy()

# %%
# Generate Categories
unique_students["Categories"] = select_categories(n_unique_students)

# %%
unique_students["Tags"] = select_tags(n_unique_students)

# %%
unique_students["Cumulative GPA"] = generate_cumulative_gpas(len(unique_students))

# %%
assigned_staff_role_probabilities_no_professor = {
    role: prob
    for role, prob in assigned_staff_role_probabilities.items()
    if role.casefold() != "professor"
}
staff_df = generate_staff_df(
    fake=fake,
    assigned_staff_role_probabilities=assigned_staff_role_probabilities_no_professor,
    n_staff=n_unique_students // 40,
    existing_emails=existing_emails,
    existing_ids=existing_ids,
)

unique_students["Assigned Staff"] = select_assigned_staff(
    staff_df, assigned_staff_role_probabilities, n_unique_students
)

# %%
existing_emails = pd.concat([existing_emails, staff_df["email"]])
existing_ids = pd.concat([existing_ids, staff_df["id"]])

# %%
num_courses_per_student = sample_from_dict(
    num_courses_per_student_distribution, size=n_unique_students
)

# %%
# Replicate each student entry based on the number of courses they're taking
replicated_students = unique_students.loc[
    unique_students.index.repeat(num_courses_per_student)
].reset_index(drop=True)

# %%
# Course info
# Generate a course schedule to assign students courses
n_course_numbers_needed = np.ceil(
    n_records / avg_students_per_section / avg_sections_per_course
).astype(int)

n_instructors_needed = np.ceil(
    n_course_numbers_needed / avg_sections_per_instructor
).astype(int)

n_sections_per_course = sample_from_dict(
    sections_per_course_distribution, size=n_course_numbers_needed
)

section_id_nums = np.random.choice(
    np.arange(10**4, 10**5), size=sum(n_sections_per_course), replace=False
)

course_depts = (
    "AAS,AHS,ANT,ART,CAN,CIN,COM,CSC,DES,EAS,EDL,EDU,ENG,FME,GEN,HIS,HSC,LAN,LAS,LAW,"
    "MED,MTH,MUS,NUR,OBG,OPH,PHA,PHE,PHY,PTH,QRM,RAC,ROM,SCI,SCW,SLV,SOC,STT,SUR,THR,"
    "URB,VIA"
).split(",")
course_number_ints = sample_from_dict(
    {
        i: 1 / i  # Higher course numbers map to lower probabilities
        for i in range(95, 601)  # Course numbers can fall between 95 and 601
    },
    size=180,
    replace=False,
)


# Add some course numbers that we definitely want to see
course_numbers_set = set(course_number_ints) | {101, 200, 600, "115L", "105H"}
course_numbers_list = sorted(
    f"{x:03}" if isinstance(x, int) else str(x) for x in course_numbers_set
)


course_numbers = (
    pd.Series(
        f"{course_dept}-"
        + (
            f"{course_number:03}"
            if isinstance(course_number, int)
            else str(course_number)
        )
        for course_dept in course_depts
        for course_number in course_numbers_list
    )
    .sample(n_course_numbers_needed)
    .sort_values(ignore_index=True)
)

course_names = pd.Series(
    fake["en_US"].unique.sentence(nb_words=6) for _ in range(n_course_numbers_needed)
).str.replace(
    r"\.$", "", regex=True  # strip the final period
)


course_schedule_list = []
for course_number, course_name, num_course_sections, section_id_num in zip(
    course_numbers, course_names, n_sections_per_course, section_id_nums
):
    for _section_num in range(num_course_sections):
        course_schedule_list.append(
            {
                "Course Number": course_number,
                "Course Name": course_name,
                "Section": section_id_num,
            }
        )

course_schedule = pd.DataFrame(course_schedule_list)

course_schedule

# %%
instructors_df = pd.DataFrame()
instructors_df["instructor_names"] = pd.Series(
    [
        f"{fake[locale].last_name()}, {fake[locale].first_name()}"
        for locale in sample_from_dict(locales_dict, size=n_instructors_needed)
    ]
)
instructors_df["instructor_emails"] = generate_emails(
    instructors_df["instructor_names"],
    existing_emails=existing_emails,
)
existing_emails = pd.concat([existing_emails, instructors_df["instructor_emails"]])

instructors_df["instructor_id"] = pd.NA
while instructors_df["instructor_id"].isna().any():
    # Repeat the ID number assignment if anyone overlaps with a Student ID or Staff ID
    # (even tho this is EXTREMELY unlikely!)
    instructors_df["instructor_id"] = [
        f"ID{rand_id:09}" if f"ID{rand_id:09}" not in existing_ids else pd.NA
        for rand_id in random.sample(range(10**9), n_instructors_needed)
    ]
existing_ids = pd.concat([existing_ids, instructors_df["instructor_id"]])


instructors_df["is_assignable_as_staff"] = np.random.choice(
    [True, False],
    size=len(instructors_df),
    p=[
        assigned_staff_role_probabilities["Professor"],
        1 - assigned_staff_role_probabilities.pop("Professor"),
    ],
)

instructors_df["name_email_formatted"] = (
    instructors_df["instructor_names"]
    + " ("
    + instructors_df["instructor_id"]
    + ") <"
    + instructors_df["instructor_emails"]
    + ">"
)

num_instructors_per_section = sample_from_dict(
    instructors_per_course_distribution, len(course_schedule)
)

course_schedule["Instructors"] = [
    (
        "; ".join(instructors_df["name_email_formatted"].sample(n).sort_values())
        if n > 0
        else None
    )
    for n in num_instructors_per_section
]

# %%
# Schedule information
# Start Date & End Date
course_schedule["Start Date"] = [
    fake.date_between(start_date="-3Months", end_date="+3Months")
    for _ in range(len(course_schedule))
]
course_schedule["End Date"] = course_schedule["Start Date"] + pd.Timedelta(weeks=10)

# %%
# Start Time & End Time
course_schedule["Start Time"] = (
    sample_from_dict(start_times_distribution, size=len(course_schedule)).astype(
        np.object_
    )
    + " CT"
)


LATEST_POSSIBLE_END_TIME = pd.to_datetime("10:00 PM", format="%I:%M %p")

potential_end_times_with_arbitrary_date = (
    pd.to_datetime(
        course_schedule["Start Time"].str.replace(" CT", ""), format="%I:%M %p"
    )
    + (
        sample_from_dict(
            course_duration_distribution, size=len(course_schedule)
        ).astype("timedelta64")
    )
).rename("Potential End Times")

potential_end_times_with_arbitrary_date.loc[
    potential_end_times_with_arbitrary_date > LATEST_POSSIBLE_END_TIME
] = LATEST_POSSIBLE_END_TIME

course_schedule["End Time"] = (
    potential_end_times_with_arbitrary_date.dt.strftime("%I:%M %p")
    .str.lstrip("0")
    .rename("End Time")
) + " CT"

# %%
# Class Days
course_schedule["Class Days"] = sample_from_dict(
    class_days_distribution, len(course_schedule)
)

# %%
# Generate the total number of courses needed by summing up courses for each student
total_courses_needed = num_courses_per_student.sum()

# Generate a random course index for each needed course
random_course_indices = np.random.choice(
    course_schedule.index, size=total_courses_needed, replace=True
)

# Assign these random course indices to each student based on how many courses they need
student_course_indices = np.split(
    random_course_indices, np.cumsum(num_courses_per_student)[:-1]
)

# Create a DataFrame for replicated students
replicated_students = pd.DataFrame(
    {
        "Student ID": np.repeat(unique_students["Student ID"], num_courses_per_student),
        "Course Index": np.concatenate(student_course_indices),
    }
)

# Merge the replicated students DataFrame with the course schedule using the course
# index. Then, drop the "Course Index" column as it's no longer needed. Finally, drop
# duplicate enrollments for each student (this behavior may change in the future).
replicated_students = (
    replicated_students.merge(
        course_schedule.reset_index(),
        left_on="Course Index",
        right_index=True,
        how="left",
    )
    .drop(columns=["Course Index"])
    .drop_duplicates(["Student ID", "Course Number"])
)

# Merge with unique_students to get student details
replicated_students = replicated_students.merge(
    unique_students, on="Student ID", how="left"
)

replicated_students

# %%
# Add Professors to "Assigned Staff"
# Explode the instructors list, since some sections will have multiple instructors
exploded_instructors = (
    replicated_students["Instructors"].str.split("; ").explode().dropna().to_frame()
)

# Add the student IDs to each row, aligning on the index
exploded_instructors["Student ID"] = replicated_students["Student ID"]

# Merge the info from the `instructors_df`
# we only care about those instructors who are assignable as staff
merged_enrollment_instructor_data = exploded_instructors.merge(
    instructors_df[instructors_df["is_assignable_as_staff"]],
    left_on="Instructors",
    right_on="name_email_formatted",
    how="inner",
)

# Use a regex to format the names how we'd like for the "Assigned Staff" column
merged_enrollment_instructor_data[
    "Assigned Professors"
] = merged_enrollment_instructor_data["instructor_names"].str.replace(
    r"^(?P<last>[\w '\-]+), (?P<first>[\w '\-]+)$",
    r"\g<first> \g<last> (Professor)",
    regex=True,
)

# Group the staff by instructors and `", ".join` them
assigned_professors = merged_enrollment_instructor_data.groupby("Student ID")[
    "Assigned Professors"
].agg(", ".join)

# Merge the assigned staff back to our enrollments report
replicated_students["Assigned Staff"] = (
    replicated_students[["Student ID", "Assigned Staff"]]
    .merge(
        assigned_professors,
        how="left",
        left_on="Student ID",
        right_index=True,
    )
    .fillna("")
    .apply(
        lambda x: x["Assigned Staff"]
        + (", " if x["Assigned Staff"] and x["Assigned Professors"] else "")
        + x["Assigned Professors"],
        axis="columns",
    )
)

# %%
# Alphabetize each student's "Assigned Staff"
replicated_students["Assigned Staff"] = (
    replicated_students["Assigned Staff"].str.split(", ").apply(sorted).apply(", ".join)
)

# %%
# Enrollment info
replicated_students["Dropped?"] = sample_from_dict(
    {"Yes": 0.24, "No": 0.76}, size=len(replicated_students)
)
replicated_students["Dropped Date"] = dropped_dates = [
    fake.date_between(start_date="-1y", end_date="today") if drop == "Yes" else None
    for drop in replicated_students["Dropped?"]
]

# %%
# Add other enrollment-specific information (e.g., grades, attendance) in a similar
# manner
replicated_students["Midterm Grade"] = np.random.choice(
    ["A", "B", "C", "D", "F"], len(replicated_students)
)
replicated_students["Final Grade"] = np.random.choice(
    ["A", "B", "C", "D", "F"], len(replicated_students)
)
replicated_students["Total Progress Reports"] = np.random.poisson(
    0.4, len(replicated_students)
)

# The proportion of students with 0 absences
PERFECT_ATTENDANCE_RATE = 0.85
# Model the distribution of non-zero absences
replicated_students["Absences"] = replicated_students["Unexcused Absences"] = (
    np.random.poisson(2.84, size=len(replicated_students)).clip(min=1)
)
# Replace `PERFECT_ATTENDANCE_RATE` proportion of records with 0 absences
replicated_students.loc[
    np.random.rand(len(replicated_students)) > PERFECT_ATTENDANCE_RATE, "Absences"
] = 0
replicated_students["Excused Absences"] = 0

# %%
# Create DataFrame
# and ensure that the dataframe is not longer than `n_records`
df_unordered_columns = replicated_students.iloc[:n_records]

# Order the columns as expected:
df = df_unordered_columns[
    [
        "Student Name",
        "Student E-mail",
        "Student ID",
        "Student Alternate ID",
        "Categories",
        "Tags",
        "Classification",
        "Major",
        "Cumulative GPA",
        "Assigned Staff",
        "Course Name",
        "Course Number",
        "Section",
        "Instructors",
        "Dropped?",
        "Dropped Date",
        "Midterm Grade",
        "Final Grade",
        "Total Progress Reports",
        "Absences",
        "Unexcused Absences",
        "Excused Absences",
        "Credit Hours",
        "Start Date",
        "End Date",
        "Start Time",
        "End Time",
        "Class Days",
    ]
].replace({None: np.NaN})

# %%
df

# %%
p = Path(EAB_tools.__file__).parent / "tests" / "io" / "data"
print(f"Saving to {p / 'campus-v2report-enrollment.csv'}")
df.to_csv(p / "campus-v2report-enrollment.csv", index=False)

# %%
with open(p / "campus-v2report-enrollment.csv", "r+", encoding="UTF-8") as f:
    today = f"{pd.Timestamp.today():%m/%d/%Y %H:%M:00}"

    lines = [
        f"Example University,Student Enrollments,,{today},Moshe Rubin\n",
        "\n",
    ] + f.readlines()

    f.seek(0)
    f.writelines(lines)
