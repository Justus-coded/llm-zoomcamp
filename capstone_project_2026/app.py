"""Flask API for the Data Roles Assistant."""

from __future__ import annotations

import os
import uuid
from typing import Any

from flask import Flask, jsonify, request

import db
from data_role_assistant.rag import rag, search


app = Flask(__name__)

FILTER_FIELDS = {
    "location",
    "company_name",
    "industry",
    "sector",
    "type_of_ownership",
    "size",
}


def _positive_int(value: Any, default: int = 10) -> int:
    """Parse a positive integer query parameter."""

    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError("num_results must be at least 1")
    return parsed


def _query_filters() -> dict[str, str]:
    """Read supported exact-match filters from query parameters."""

    return {
        field: request.args[field]
        for field in FILTER_FIELDS
        if request.args.get(field)
    }


def _save_question(
    question: str,
    model: str | None = None,
    evaluation_model: str | None = None,
    filter_dict: dict[str, Any] | None = None,
    num_results: int = 10,
) -> tuple[str, dict[str, Any]]:
    """Run RAG and persist the resulting conversation."""

    conversation_id = str(uuid.uuid4())
    answer_data = rag(
        query=question,
        model=model,
        evaluation_model=evaluation_model,
        filter_dict=filter_dict,
        num_results=num_results,
    )
    db.save_conversation(
        conversation_id=conversation_id,
        question=question,
        answer_data=answer_data,
    )
    return conversation_id, answer_data


@app.get("/health")
def health() -> tuple[Any, int]:
    """Return a lightweight service health response."""

    return jsonify({"status": "ok"}), 200


@app.get("/search")
def search_endpoint() -> tuple[Any, int]:
    """Search job postings using the boosted basic retriever."""

    query = request.args.get("q") or request.args.get("query", "")
    if not query.strip():
        return jsonify({"error": "Provide a q or query parameter"}), 400

    try:
        num_results = _positive_int(request.args.get("num_results"))
        results = search(
            query=query,
            filter_dict=_query_filters(),
            num_results=num_results,
        )
        return jsonify(
            {
                "query": query,
                "count": len(results),
                "results": results,
            }
        ), 200
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/question")
def handle_question() -> tuple[Any, int]:
    """Answer and persist a user question."""

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "No question provided"}), 400

    filter_dict = payload.get("filter_dict") or {}
    if not isinstance(filter_dict, dict):
        return jsonify({"error": "filter_dict must be an object"}), 400

    try:
        conversation_id, answer_data = _save_question(
            question=question,
            model=payload.get("model"),
            evaluation_model=payload.get("evaluation_model"),
            filter_dict=filter_dict,
            num_results=_positive_int(payload.get("num_results")),
        )
        return jsonify(
            {
                "conversation_id": conversation_id,
                "question": question,
                **answer_data,
            }
        ), 200
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Failed to process question")
        return jsonify({"error": "Failed to process question"}), 503


@app.post("/rag")
def rag_endpoint() -> tuple[Any, int]:
    """Backward-compatible alias for the conversation endpoint."""

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    question = payload.get("question") or payload.get("query", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "No question provided"}), 400

    filter_dict = payload.get("filter_dict") or {}
    if not isinstance(filter_dict, dict):
        return jsonify({"error": "filter_dict must be an object"}), 400

    try:
        conversation_id, answer_data = _save_question(
            question=question,
            model=payload.get("model"),
            evaluation_model=payload.get("evaluation_model"),
            filter_dict=filter_dict,
            num_results=_positive_int(payload.get("num_results")),
        )
        return jsonify(
            {
                "conversation_id": conversation_id,
                "question": question,
                **answer_data,
            }
        ), 200
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Failed to process RAG request")
        return jsonify({"error": "Failed to process RAG request"}), 503


@app.post("/feedback")
def handle_feedback() -> tuple[Any, int]:
    """Persist positive or negative feedback for a conversation."""

    payload = request.get_json(silent=True) or {}
    conversation_id = payload.get("conversation_id")
    feedback = payload.get("feedback")

    if (
        not isinstance(conversation_id, str)
        or not conversation_id
        or isinstance(feedback, bool)
        or feedback not in (1, -1)
    ):
        return jsonify({"error": "Invalid input"}), 400

    try:
        db.save_feedback(
            conversation_id=conversation_id,
            feedback=feedback,
        )
        return jsonify(
            {"message": f"Feedback received: {feedback}"}
        ), 200
    except Exception:
        app.logger.exception("Failed to save feedback")
        return jsonify({"error": "Failed to save feedback"}), 503


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
