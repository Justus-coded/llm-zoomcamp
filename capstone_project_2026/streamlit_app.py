"""Streamlit interface for testing RAG and monitoring conversations."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pandas as pd
import streamlit as st

import db
from data_role_assistant.rag import DEFAULT_MODEL, rag


st.set_page_config(
    page_title="Data Roles Assistant",
    page_icon="📊",
    layout="wide",
)


def _filters_from_sidebar() -> dict[str, str]:
    """Build exact-match minsearch filters from optional sidebar inputs."""

    filters: dict[str, str] = {}
    for field, label in (
        ("location", "Location"),
        ("industry", "Industry"),
        ("company_name", "Company"),
    ):
        value = st.sidebar.text_input(
            f"{label} filter",
            placeholder=f"Exact {label.lower()} value",
        ).strip()
        if value:
            filters[field] = value
    return filters


def _save_response(
    question: str,
    answer_data: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Persist a response and return (conversation id, database error)."""

    conversation_id = str(uuid.uuid4())
    try:
        db.save_conversation(
            conversation_id=conversation_id,
            question=question,
            answer_data=answer_data,
        )
        return conversation_id, None
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=10)
def load_dashboard_data() -> dict[str, Any]:
    """Load monitoring data with a short cache to reduce database traffic."""

    return db.get_dashboard_data()


def render_ask_page(filters: dict[str, str]) -> None:
    """Render the RAG testing and feedback page."""

    st.title("Data Roles Assistant")
    st.write(
        "Ask questions about Business Analyst, Data Analyst, "
        "Data Engineer, and Data Scientist job postings."
    )

    with st.form("question_form"):
        question = st.text_area(
            "Question",
            placeholder="Find data analyst jobs in healthcare near New York",
            height=100,
        )
        left, right = st.columns(2)
        with left:
            model = st.text_input(
                "Answer model",
                value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            )
        with right:
            evaluation_model = st.text_input(
                "Evaluation model",
                value=os.getenv(
                    "OPENAI_EVALUATION_MODEL",
                    "gpt-5.6-luna",
                ),
            )
        num_results = st.slider(
            "Retrieved jobs",
            min_value=1,
            max_value=20,
            value=10,
        )
        submitted = st.form_submit_button(
            "Ask",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a question before submitting.")
            return

        with st.spinner(
            "Retrieving jobs, generating an answer, and evaluating it..."
        ):
            try:
                answer_data = rag(
                    query=question,
                    model=model.strip() or None,
                    evaluation_model=evaluation_model.strip() or None,
                    filter_dict=filters,
                    num_results=num_results,
                )
            except Exception as exc:
                st.error(f"RAG request failed: {exc}")
                return

        conversation_id, database_error = _save_response(
            question=question,
            answer_data=answer_data,
        )
        st.session_state["latest_question"] = question
        st.session_state["latest_answer"] = answer_data
        st.session_state["latest_conversation_id"] = conversation_id
        st.session_state["latest_database_error"] = database_error

    answer_data = st.session_state.get("latest_answer")
    if not answer_data:
        return

    conversation_id = st.session_state.get("latest_conversation_id")
    database_error = st.session_state.get("latest_database_error")

    st.subheader("Answer")
    st.write(answer_data["answer"])

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Relevance",
        answer_data["relevance"],
    )
    metric_columns[1].metric(
        "Response time",
        f"{answer_data['response_time']:.2f}s",
    )
    metric_columns[2].metric(
        "Estimated cost",
        f"${answer_data['openai_cost']:.5f}",
    )
    metric_columns[3].metric(
        "Total tokens",
        f"{answer_data['total_tokens']:,}",
    )

    with st.expander("Evaluator explanation"):
        st.write(answer_data["relevance_explanation"])

    if database_error:
        st.warning(
            "The answer was generated, but it was not saved. "
            f"Is PostgreSQL running? ({database_error})"
        )
        return

    st.caption(f"Conversation ID: `{conversation_id}`")
    st.subheader("Was this answer useful?")
    feedback_left, feedback_right = st.columns(2)

    with feedback_left:
        if st.button(
            "👍 Helpful",
            key=f"positive-{conversation_id}",
            use_container_width=True,
        ):
            try:
                db.save_feedback(conversation_id, 1)
                st.success("Positive feedback saved.")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")

    with feedback_right:
        if st.button(
            "👎 Not helpful",
            key=f"negative-{conversation_id}",
            use_container_width=True,
        ):
            try:
                db.save_feedback(conversation_id, -1)
                st.success("Negative feedback saved.")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")


def render_dashboard_page() -> None:
    """Render database-backed feedback and service monitoring."""

    st.title("Monitoring dashboard")
    st.caption(
        "Conversation quality, latency, estimated cost, and user feedback."
    )

    if st.button("Refresh dashboard"):
        load_dashboard_data.clear()
        st.rerun()

    try:
        dashboard = load_dashboard_data()
    except Exception as exc:
        st.warning(
            "The dashboard cannot connect to PostgreSQL. Start the stack "
            "with `docker compose up -d` and initialize the database with "
            "`docker compose exec app python -c "
            '"import db; db.init_db()"`.'
        )
        with st.expander("Database error"):
            st.code(str(exc))
        return

    summary = dashboard["summary"]
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Conversations",
        f"{summary['total_conversations']:,}",
    )
    metric_columns[1].metric(
        "Average latency",
        f"{summary['average_response_time']:.2f}s",
    )
    metric_columns[2].metric(
        "Estimated cost",
        f"${summary['total_openai_cost']:.5f}",
    )
    metric_columns[3].metric(
        "Average tokens",
        f"{summary['average_total_tokens']:,.0f}",
    )

    timeline = pd.DataFrame(dashboard["timeline"])
    if not timeline.empty:
        timeline["timestamp"] = pd.to_datetime(timeline["timestamp"])
        timeline = timeline.set_index("timestamp")

        st.subheader("System monitoring")
        st.line_chart(
            timeline[
                ["conversations", "average_response_time"]
            ],
            use_container_width=True,
        )
        st.subheader("Estimated OpenAI cost")
        st.area_chart(
            timeline[["cost"]],
            use_container_width=True,
        )

    relevance_column, feedback_column = st.columns(2)

    with relevance_column:
        st.subheader("LLM relevance")
        relevance = pd.DataFrame(dashboard["relevance"])
        if relevance.empty:
            st.info("No evaluated conversations yet.")
        else:
            st.bar_chart(
                relevance.set_index("relevance")["count"],
                use_container_width=True,
            )

    with feedback_column:
        st.subheader("User feedback")
        feedback = pd.DataFrame(dashboard["feedback"])
        if feedback.empty:
            st.info("No feedback submitted yet.")
        else:
            st.bar_chart(
                feedback.set_index("feedback")["count"],
                use_container_width=True,
            )

    st.subheader("Recent conversations")
    recent = pd.DataFrame(dashboard["recent_conversations"])
    if recent.empty:
        st.info("No conversations recorded yet.")
    else:
        st.dataframe(
            recent,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="YYYY-MM-DD HH:mm:ss",
                ),
                "response_time": st.column_config.NumberColumn(
                    "Latency (s)",
                    format="%.2f",
                ),
                "openai_cost": st.column_config.NumberColumn(
                    "Cost",
                    format="$%.5f",
                ),
            },
            column_order=[
                "timestamp",
                "question",
                "model_used",
                "relevance",
                "response_time",
                "openai_cost",
                "feedback_count",
            ],
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    """Run the Streamlit application."""

    st.sidebar.title("Data Roles Assistant")
    st.sidebar.caption("RAG testing and operational monitoring")
    page = st.sidebar.radio(
        "View",
        ["Ask", "Dashboard"],
    )

    if page == "Ask":
        render_ask_page(_filters_from_sidebar())
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
