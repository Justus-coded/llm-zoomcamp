"""Search and retrieval-augmented generation for data-role job postings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from minsearch import Index
from openai import OpenAI

from . import ingest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
EVALUATION_MODEL = os.getenv("OPENAI_EVALUATION_MODEL", "gpt-5.6-luna")

BOOST = {
    "job_title": 3.63,
    "job_description": 1.61,
    "location": 0.75,
    "industry": 2.03,
    "company_name": 2.01,
    "sector": 0.01,
    "salary_estimate": 0.99,
}

_index: Index | None = None
_openai_client: OpenAI | None = None


def get_index() -> Index:
    """Return the cached search index, loading it on first use."""

    global _index
    if _index is None:
        _index = ingest.load_index()
    return _index


def get_openai_client() -> OpenAI:
    """Return the OpenAI client, creating it on first LLM request."""

    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def search(
    query: str,
    filter_dict: dict[str, Any] | None = None,
    boost: dict[str, float] | None = None,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Search job postings with boosted field-level minsearch retrieval."""

    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if num_results < 1:
        raise ValueError("num_results must be at least 1")

    return get_index().search(
        query=query,
        filter_dict=filter_dict or {},
        boost_dict=boost if boost is not None else BOOST,
        num_results=num_results,
    )


def search_b(
    query: str,
    filter_dict: dict[str, Any] | None = None,
    num_results: int = 10,
) -> list[dict[str, Any]]:
    """Backward-compatible name for the boosted basic search."""

    return search(
        query=query,
        filter_dict=filter_dict,
        boost=BOOST,
        num_results=num_results,
    )


PROMPT_TEMPLATE = """
You are a data-career search assistant. Answer the QUESTION using only the facts in the CONTEXT from the job database.

Rules:
- Do not invent information that is not in the CONTEXT.
- If the answer is not available, say that it is not listed.
- When recommending jobs, include the job title, company, location, salary estimate, and a short reason.
- Do not assume remote work, qualifications, benefits, or responsibilities unless explicitly stated.
- If multiple jobs are relevant, compare them clearly.

QUESTION:
{question}

CONTEXT:
{context}
""".strip()

ENTRY_TEMPLATE = """
Job ID: {id}
Job Title: {job_title}
Salary Estimate: {salary_estimate}
Company Name: {company_name}
Location: {location}
Headquarters: {headquarters}
Company Size: {size}
Type of Ownership: {type_of_ownership}
Industry: {industry}
Sector: {sector}
Competitors: {competitors}

Job Description:
{job_description}
""".strip()

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the question.
Classify it as exactly one of: NON_RELEVANT, PARTLY_RELEVANT, or RELEVANT.

Question:
{question}

Generated Answer:
{answer}

Return only valid JSON without Markdown code fences:
{{
  "Relevance": "NON_RELEVANT",
  "Explanation": "Briefly explain the classification."
}}
""".strip()


def get_value(document: dict[str, Any], field: str) -> str:
    """Return a display-safe value for a job field."""

    value = document.get(field)
    if value is None or str(value).strip() in {"", "-1"}:
        return "Not listed"
    return str(value).strip()


def build_prompt(
    query: str,
    search_results: list[dict[str, Any]],
) -> str:
    """Build a grounded prompt from retrieved job postings."""

    context_entries = [
        ENTRY_TEMPLATE.format(
            id=get_value(document, "id"),
            job_title=get_value(document, "job_title"),
            salary_estimate=get_value(document, "salary_estimate"),
            company_name=get_value(document, "company_name"),
            location=get_value(document, "location"),
            headquarters=get_value(document, "headquarters"),
            size=get_value(document, "size"),
            type_of_ownership=get_value(document, "type_of_ownership"),
            industry=get_value(document, "industry"),
            sector=get_value(document, "sector"),
            competitors=get_value(document, "competitors"),
            job_description=get_value(document, "job_description"),
        )
        for document in search_results
    ]

    context = "\n\n---\n\n".join(context_entries)
    return PROMPT_TEMPLATE.format(
        question=query,
        context=context or "No matching job listings were found.",
    ).strip()


def evaluate_relevance(
    question: str,
    answer: str,
    model: str | None = None,
) -> tuple[dict[str, str], dict[str, int]]:
    """Evaluate an answer with an LLM-as-a-judge."""

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
    )
    evaluation_text, token_stats = llm(
        prompt,
        model=model or EVALUATION_MODEL,
    )

    try:
        cleaned = evaluation_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1)
            cleaned = cleaned.removesuffix("```").strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Evaluation response was not a JSON object")
        return {
            "Relevance": str(parsed.get("Relevance", "UNKNOWN")),
            "Explanation": str(
                parsed.get("Explanation", "No explanation provided")
            ),
        }, token_stats
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "Relevance": "UNKNOWN",
            "Explanation": "Failed to parse evaluation",
        }, token_stats


def calculate_openai_cost(
    model: str,
    tokens: dict[str, int],
) -> float:
    """Estimate model cost using the configured GPT-5 rates per million tokens."""

    if "gpt-5.6-luna" in model or "gpt-5.4-luna" in model:
        return (
            tokens["prompt_tokens"] * 0.15
            + tokens["completion_tokens"] * 0.60
        ) / 1_000_000
    return 0.0


def llm(
    prompt: str,
    model: str | None = None,
) -> tuple[str, dict[str, int]]:
    """Generate an answer and return its token usage."""

    selected_model = model or DEFAULT_MODEL
    response = get_openai_client().responses.create(
        model=selected_model,
        input=[{"role": "user", "content": prompt}],
    )

    usage = getattr(response, "usage", None)
    token_stats = {
        "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return response.output_text, token_stats


def rag(
    query: str,
    model: str | None = None,
    evaluation_model: str | None = None,
    filter_dict: dict[str, Any] | None = None,
    num_results: int = 10,
) -> dict[str, Any]:
    """Retrieve, answer, evaluate, and return usage/cost metadata."""

    selected_model = model or DEFAULT_MODEL
    selected_evaluation_model = evaluation_model or EVALUATION_MODEL
    started_at = perf_counter()
    search_results = search(
        query=query,
        filter_dict=filter_dict,
        num_results=num_results,
    )
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=selected_model)
    relevance, evaluation_token_stats = evaluate_relevance(
        question=query,
        answer=answer,
        model=selected_evaluation_model,
    )

    return {
        "answer": answer,
        "model_used": selected_model,
        "response_time": perf_counter() - started_at,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation",
            "Failed to parse evaluation",
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": evaluation_token_stats["prompt_tokens"],
        "eval_completion_tokens": evaluation_token_stats["completion_tokens"],
        "eval_total_tokens": evaluation_token_stats["total_tokens"],
        "openai_cost": (
            calculate_openai_cost(selected_model, token_stats)
            + calculate_openai_cost(
                selected_evaluation_model,
                evaluation_token_stats,
            )
        ),
    }
