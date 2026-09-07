"""Provision the PostgreSQL datasource and dashboard in Grafana."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
DATASOURCE_NAME = os.getenv(
    "GRAFANA_DATASOURCE_NAME",
    "Data Roles PostgreSQL",
)


def wait_for_grafana(
    session: requests.Session,
    attempts: int = 60,
    delay: int = 2,
) -> None:
    """Wait until Grafana's health endpoint responds."""

    for _ in range(attempts):
        try:
            response = session.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    raise RuntimeError("Grafana did not become ready in time")


def create_or_update_datasource(session: requests.Session) -> str:
    """Create the project PostgreSQL datasource and return its UID."""

    datasource = {
        "name": DATASOURCE_NAME,
        "type": "postgres",
        "access": "proxy",
        "url": os.getenv("GRAFANA_POSTGRES_URL", "postgres:5432"),
        "user": os.getenv("POSTGRES_USER", "user"),
        "database": os.getenv("POSTGRES_DB", "data_role_assistant"),
        "basicAuth": False,
        "isDefault": False,
        "jsonData": {
            "database": os.getenv("POSTGRES_DB", "data_role_assistant"),
            "postgresVersion": 16,
            "sslmode": "disable",
            "timescaledb": False,
        },
        "secureJsonData": {
            "password": os.getenv("POSTGRES_PASSWORD", "password"),
        },
    }

    lookup_url = (
        f"{GRAFANA_URL}/api/datasources/name/{quote(DATASOURCE_NAME)}"
    )
    existing = session.get(lookup_url, timeout=10)

    if existing.status_code == 200:
        current = existing.json()
        datasource["id"] = current["id"]
        datasource["uid"] = current.get("uid")
        response = session.put(
            f"{GRAFANA_URL}/api/datasources/{current['id']}",
            json=datasource,
            timeout=10,
        )
    elif existing.status_code == 404:
        response = session.post(
            f"{GRAFANA_URL}/api/datasources",
            json=datasource,
            timeout=10,
        )
    else:
        existing.raise_for_status()
        raise RuntimeError("Unable to inspect the Grafana datasource")

    response.raise_for_status()
    response_data = response.json()
    return (
        response_data.get("datasource", {}).get("uid")
        or response_data.get("uid")
        or datasource.get("uid")
    )


def replace_datasource_uid(value, datasource_uid: str):
    """Replace the dashboard datasource placeholder recursively."""

    placeholder = "${POSTGRES_DATASOURCE_UID}"
    if isinstance(value, dict):
        return {
            key: replace_datasource_uid(item, datasource_uid)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            replace_datasource_uid(item, datasource_uid)
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(placeholder, datasource_uid)
    return value


def provision_dashboard(
    session: requests.Session,
    datasource_uid: str,
) -> None:
    """Load and publish the project dashboard JSON."""

    dashboard_path = Path(__file__).with_name("dashboard.json")
    dashboard = json.loads(dashboard_path.read_text())
    dashboard = replace_datasource_uid(dashboard, datasource_uid)

    response = session.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        json={
            "dashboard": dashboard,
            "folderId": 0,
            "overwrite": True,
        },
        timeout=10,
    )
    response.raise_for_status()


def main() -> None:
    with requests.Session() as session:
        session.auth = (GRAFANA_USER, GRAFANA_PASSWORD)
        wait_for_grafana(session)
        datasource_uid = create_or_update_datasource(session)
        if not datasource_uid:
            raise RuntimeError("Grafana did not return a datasource UID")
        provision_dashboard(session, datasource_uid)

    print("Grafana datasource and dashboard provisioned successfully")


if __name__ == "__main__":
    main()
