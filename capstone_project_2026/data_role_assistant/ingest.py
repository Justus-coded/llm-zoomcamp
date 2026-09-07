"""Load the job dataset and build the minsearch index."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from minsearch import Index


DATA_PATH = os.getenv("DATA_PATH", "data_jobs_pydantic.csv")

TEXT_FIELDS = [
    "job_title",
    "job_description",
    "company_name",
    "location",
    "industry",
    "sector",
    "salary_estimate",
]

KEYWORD_FIELDS = [
    "id",
    "location",
    "company_name",
    "industry",
    "sector",
    "type_of_ownership",
    "size",
]


def load_documents(
    data_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load job postings as dictionaries without dropping the ``id`` field."""

    path = Path(data_path or os.getenv("DATA_PATH", DATA_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Set DATA_PATH to the CSV location."
        )

    df = pd.read_csv(path, keep_default_na=False)
    required_columns = set(TEXT_FIELDS) | set(KEYWORD_FIELDS)
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    return df.to_dict(orient="records")


def load_index(
    data_path: str | os.PathLike[str] | None = None,
) -> Index:
    """Load the job postings and return a fitted minsearch index."""

    documents = load_documents(data_path)
    index = Index(
        text_fields=TEXT_FIELDS,
        keyword_fields=KEYWORD_FIELDS,
    )
    index.fit(documents)
    return index
