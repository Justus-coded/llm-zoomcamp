"""Build the browser-friendly dataset used by the static dashboard.

Run from the project root:

    python Dashboard/build_data.py

The generated file intentionally excludes the source ``id`` and ``rating``
fields. Hourly estimates are annualized at 2,080 working hours so the salary
filter compares all listings on one scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data_jobs_pydantic.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "jobs.js"
ANNUAL_WORK_HOURS = 2_080

STATE_NAMES = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

SALARY_TOKEN = re.compile(r"(?<!\w)(\d+(?:\.\d+)?)\s*([Kk])?")
JOB_ID_LABEL = re.compile(
    r"\bjob\s*id(?:\W|Â)*[A-Za-z0-9][A-Za-z0-9_-]*",
    re.IGNORECASE,
)
RATING_LABEL = re.compile(
    r"\brating\s*[:#-]?\s*[-+]?\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def clean_value(value: str | None) -> str:
    """Return a readable value for an empty or sentinel source value."""

    value = " ".join((value or "").split())
    return "Not listed" if not value or value == "-1" else value


def parse_location(value: str | None) -> tuple[str, str, str]:
    """Split a location into city/area, state name, and state code."""

    location = clean_value(value)
    if location == "Not listed":
        return location, location, location

    parts = [part.strip() for part in location.split(",") if part.strip()]
    if len(parts) < 2:
        return location, location, location

    state_code = parts[-1].upper()
    state_name = STATE_NAMES.get(state_code, state_code)
    city = ", ".join(parts[:-1])
    return city, state_name, state_code


def parse_salary(value: str | None) -> tuple[int | None, int | None, str]:
    """Return normalized annual minimum, maximum, and basis."""

    raw = (value or "").strip()
    if not raw or raw == "-1":
        return None, None, "not-listed"

    matches = SALARY_TOKEN.findall(raw)
    if len(matches) < 2:
        return None, None, "not-listed"

    is_hourly = "hour" in raw.lower()
    numbers = []
    for number, suffix in matches[:2]:
        multiplier = 1_000 if suffix else (ANNUAL_WORK_HOURS if is_hourly else 1)
        numbers.append(round(float(number) * multiplier))

    return min(numbers), max(numbers), "hourly-annualized" if is_hourly else "annual"


def clean_description(value: str | None) -> str:
    """Normalize a description and remove excluded source metadata labels."""

    description = " ".join((value or "").split())
    description = JOB_ID_LABEL.sub("", description)
    description = RATING_LABEL.sub("", description)
    return " ".join(description.split())


def transform_row(row: dict[str, str]) -> dict[str, Any]:
    """Select and normalize the fields used by the dashboard."""

    city, state_name, state_code = parse_location(row.get("location"))
    salary_min, salary_max, salary_basis = parse_salary(
        row.get("salary_estimate")
    )
    return {
        "job_title": clean_value(row.get("job_title")),
        "company_name": clean_value(row.get("company_name")),
        "city": city,
        "state_name": state_name,
        "state_code": state_code,
        "size": clean_value(row.get("size")),
        "ownership": clean_value(row.get("type_of_ownership")),
        "industry": clean_value(row.get("industry")),
        "sector": clean_value(row.get("sector")),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_basis": salary_basis,
        "description": clean_description(row.get("job_description")),
    }


def build_dataset(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        return [transform_row(row) for row in csv.DictReader(source)]


def write_javascript(output_path: Path, records: list[dict[str, Any]]) -> None:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output_path.write_text(
        "// Generated by Dashboard/build_data.py. Do not edit by hand.\n"
        f"window.DASHBOARD_JOBS = {payload};\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = build_dataset(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_javascript(args.output, records)
    print(f"Wrote {len(records):,} records to {args.output}")


if __name__ == "__main__":
    main()
