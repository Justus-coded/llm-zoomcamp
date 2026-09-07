"""Validate and normalize the data-roles CSV with Pydantic.

The source CSV contains one ``page_content`` column.  This module extracts
the labeled values from that text, validates each record with Pydantic, and
writes a tabular CSV that can be loaded directly into pandas.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


class DataJob(BaseModel):
    """A validated, structured data-role job posting."""

    id: str = Field(description="Stable identifier based on the source row")
    job_title: str = Field(description="Title of the advertised role")
    salary_estimate: str = Field(description="Salary range shown in the source")
    job_description: str = Field(description="Full job description")
    rating: Optional[float] = Field(
        default=None,
        description="Company rating when the source contains a scalar value",
    )
    company_name: str = Field(description="Name of the hiring company")
    location: str = Field(description="Job location")
    headquarters: str = Field(description="Company headquarters")
    size: str = Field(description="Company size")
    type_of_ownership: str = Field(description="Company ownership type")
    industry: str = Field(description="Company industry")
    sector: str = Field(description="Company sector")
    competitors: str = Field(description="Competitors listed in the source")
    page_content: str = Field(
        description="Normalized labeled text generated from the structured fields"
    )


class DataJobsDataset(BaseModel):
    """Pydantic container matching the dataset shape in the example."""

    jobs: list[DataJob]


_HEADER_RE = re.compile(
    r"^Job Title:\s*(?P<job_title>.*?)\n"
    r"Salary Estimate:\s*(?P<salary_estimate>.*?)\n"
    r"Job Description:\s*",
    flags=re.DOTALL,
)

_METADATA_RE = re.compile(
    r"\nCompany Name:\s*(?P<company_name>.*?)\n"
    r"Location:\s*(?P<location>.*?)\n"
    r"Headquarters:\s*(?P<headquarters>.*?)\n"
    r"Size:\s*(?P<size>.*?)\n"
    r"Type of ownership:\s*(?P<type_of_ownership>.*?)\n"
    r"Industry:\s*(?P<industry>.*?)\n"
    r"Sector:\s*(?P<sector>.*?)\n"
    r"Competitors:\s*(?P<competitors>.*?)\s*\Z",
    flags=re.DOTALL,
)

_SCALAR_RATING_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+)$")


def _model_validate(model: type[BaseModel], values: dict[str, Any]) -> BaseModel:
    """Use the Pydantic v2 API while remaining compatible with v1."""

    if hasattr(model, "model_validate"):
        return model.model_validate(values)
    return model.parse_obj(values)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    """Use the Pydantic v2 API while remaining compatible with v1."""

    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _clean(value: str) -> str:
    """Normalize whitespace around a parsed field without changing its body."""

    return value.strip()


def _parse_rating(raw_rating: str) -> Optional[float]:
    """Return a scalar rating, or None for the malformed serialized Series."""

    value = _clean(raw_rating)
    if not _SCALAR_RATING_RE.fullmatch(value):
        return None
    return float(value)


def _build_page_content(values: dict[str, Any]) -> str:
    """Build a clean page-content value from validated field values."""

    rating = values["rating"]
    rating_text = "Not available" if rating is None else f"{rating:g}"
    return "\n".join(
        [
            f"Job Title: {values['job_title']}",
            f"Salary Estimate: {values['salary_estimate']}",
            f"Job Description: {values['job_description']}",
            f"Rating: {rating_text}",
            f"Company Name: {values['company_name']}",
            f"Location: {values['location']}",
            f"Headquarters: {values['headquarters']}",
            f"Size: {values['size']}",
            f"Type of ownership: {values['type_of_ownership']}",
            f"Industry: {values['industry']}",
            f"Sector: {values['sector']}",
            f"Competitors: {values['competitors']}",
        ]
    )


def parse_page_content(page_content: str, source_row: int) -> DataJob:
    """Parse one source page-content value into a validated ``DataJob``."""

    text = page_content.replace("\r\n", "\n").strip()
    header = _HEADER_RE.match(text)
    if header is None:
        raise ValueError("missing Job Title, Salary Estimate, or Job Description")

    # Some descriptions contain their own "Company Name:" and "Location:"
    # lines.  The final occurrence is the actual metadata block assembled by
    # the source notebook.
    metadata_start = text.rfind("\nCompany Name:")
    metadata = (
        _METADATA_RE.match(text, metadata_start)
        if metadata_start != -1
        else None
    )
    if metadata is None:
        raise ValueError("missing one or more company metadata fields")

    # Search backwards so a "Rating:" line inside a job description does not
    # become the field separator.
    rating_start = text.rfind("\nRating:", header.end(), metadata.start())
    if rating_start == -1:
        raise ValueError("missing Rating field separator")

    values: dict[str, Any] = {
        "id": f"job-{source_row:05d}",
        "job_title": _clean(header.group("job_title")),
        "salary_estimate": _clean(header.group("salary_estimate")),
        "job_description": _clean(text[header.end() : rating_start]),
        "rating": _parse_rating(text[rating_start + len("\nRating:") : metadata.start()]),
    }
    values.update(
        {
            name: _clean(value)
            for name, value in metadata.groupdict().items()
        }
    )
    values["page_content"] = _build_page_content(values)

    return _model_validate(DataJob, values)  # type: ignore[return-value]


def _default_input_path(folder: Path) -> Path:
    """Find the requested filename, allowing for the available trailing-underscore name."""

    for filename in ("data_jobs.csv", "data_jobs_.csv"):
        candidate = folder / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find data_jobs.csv or data_jobs_.csv in {folder}"
    )


def format_data_jobs(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    """Convert the source CSV and return (written, blank, malformed) counts."""

    source_df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )
    if "page_content" not in source_df.columns:
        raise ValueError("The input CSV must contain a page_content column")

    jobs: list[DataJob] = []
    blank_rows = 0
    errors: list[str] = []

    for source_row, page_content in enumerate(source_df["page_content"], start=1):
        if not page_content.strip():
            blank_rows += 1
            continue
        try:
            jobs.append(parse_page_content(page_content, source_row))
        except (ValueError, ValidationError) as exc:
            errors.append(f"source row {source_row}: {exc}")

    if errors:
        details = "\n".join(errors[:10])
        suffix = "" if len(errors) <= 10 else f"\n... and {len(errors) - 10} more"
        raise ValueError(f"Could not parse {len(errors)} source rows:\n{details}{suffix}")

    dataset = _model_validate(DataJobsDataset, {"jobs": jobs})
    output_df = pd.DataFrame([_model_dump(job) for job in dataset.jobs])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return len(jobs), blank_rows, sum(job.rating is None for job in jobs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validated, structured data-jobs CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input CSV; defaults to data_jobs.csv or data_jobs_.csv in this folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV; defaults to data_jobs_pydantic.csv in this folder",
    )
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    input_path = args.input or _default_input_path(folder)
    output_path = args.output or folder / "data_jobs_pydantic.csv"
    written, blank, missing_ratings = format_data_jobs(input_path, output_path)
    print(f"Wrote {written:,} validated jobs to {output_path}")
    print(f"Skipped {blank:,} blank source rows")
    print(f"Ratings unavailable because the source value was not scalar: {missing_ratings:,}")


if __name__ == "__main__":
    main()
