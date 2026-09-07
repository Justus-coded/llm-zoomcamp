"""PostgreSQL persistence for conversations and user feedback."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor


load_dotenv(Path(__file__).resolve().parent / ".env")

TZ_INFO = os.getenv("TZ", "America/Chicago")
tz = ZoneInfo(TZ_INFO)


def get_db_connection():
    """Create a PostgreSQL connection from environment variables."""

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "data_role_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        cursor_factory=DictCursor,
    )


def init_db(reset: bool = False) -> None:
    """Create the application tables.

    ``reset=True`` drops the existing application tables before recreating
    them. The default is non-destructive and is suitable for deployments.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if reset:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    response_time DOUBLE PRECISION NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost DOUBLE PRECISION NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id),
                    feedback INTEGER NOT NULL
                        CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
                """
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_conversation(
    conversation_id: str,
    question: str,
    answer_data: dict[str, Any],
    timestamp: datetime | None = None,
) -> None:
    """Persist one RAG response and its evaluation metadata."""

    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    id, question, answer, model_used, response_time,
                    relevance, relevance_explanation, prompt_tokens,
                    completion_tokens, total_tokens, eval_prompt_tokens,
                    eval_completion_tokens, eval_total_tokens, openai_cost,
                    timestamp
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["model_used"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_feedback(
    conversation_id: str,
    feedback: int,
    timestamp: datetime | None = None,
) -> None:
    """Persist positive or negative feedback for a conversation."""

    if feedback not in (-1, 1):
        raise ValueError("feedback must be either 1 or -1")
    if not conversation_id:
        raise ValueError("conversation_id must not be empty")

    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, feedback, timestamp
                )
                VALUES (%s, %s, %s)
                """,
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _rows_as_dicts(rows) -> list[dict[str, Any]]:
    """Convert psycopg2 DictRows into regular dictionaries."""

    return [dict(row) for row in rows]


def get_dashboard_data(
    recent_limit: int = 50,
) -> dict[str, Any]:
    """Return conversation, relevance, feedback, and monitoring metrics."""

    if recent_limit < 1:
        raise ValueError("recent_limit must be at least 1")

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)::INTEGER AS total_conversations,
                    COALESCE(AVG(response_time), 0)::DOUBLE PRECISION
                        AS average_response_time,
                    COALESCE(SUM(openai_cost), 0)::DOUBLE PRECISION
                        AS total_openai_cost,
                    COALESCE(AVG(total_tokens), 0)::DOUBLE PRECISION
                        AS average_total_tokens
                FROM conversations
                """
            )
            summary = dict(cur.fetchone())

            cur.execute(
                """
                SELECT relevance, COUNT(*)::INTEGER AS count
                FROM conversations
                GROUP BY relevance
                ORDER BY count DESC
                """
            )
            relevance = _rows_as_dicts(cur.fetchall())

            cur.execute(
                """
                SELECT
                    CASE WHEN feedback = 1 THEN 'Positive'
                         ELSE 'Negative'
                    END AS feedback,
                    COUNT(*)::INTEGER AS count
                FROM feedback
                GROUP BY 1
                ORDER BY 1
                """
            )
            feedback = _rows_as_dicts(cur.fetchall())

            cur.execute(
                """
                SELECT
                    date_trunc('hour', timestamp) AS timestamp,
                    COUNT(*)::INTEGER AS conversations,
                    COALESCE(SUM(openai_cost), 0)::DOUBLE PRECISION AS cost,
                    COALESCE(AVG(response_time), 0)::DOUBLE PRECISION
                        AS average_response_time
                FROM conversations
                GROUP BY 1
                ORDER BY 1
                """
            )
            timeline = _rows_as_dicts(cur.fetchall())

            cur.execute(
                """
                SELECT
                    c.id,
                    c.question,
                    c.model_used,
                    c.response_time,
                    c.relevance,
                    c.openai_cost,
                    c.timestamp,
                    COUNT(f.id)::INTEGER AS feedback_count
                FROM conversations AS c
                LEFT JOIN feedback AS f ON f.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.timestamp DESC
                LIMIT %s
                """,
                (recent_limit,),
            )
            recent_conversations = _rows_as_dicts(cur.fetchall())

        return {
            "summary": summary,
            "relevance": relevance,
            "feedback": feedback,
            "timeline": timeline,
            "recent_conversations": recent_conversations,
        }
    finally:
        conn.close()

