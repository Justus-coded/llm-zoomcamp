# Data Roles Dashboard

This is a standalone HTML dashboard built from `data_jobs_pydantic.csv`.
It excludes the source `id` and `rating` fields.

## Build the dashboard data

Run this from the project root whenever the CSV changes:

```bash
python Dashboard/build_data.py
```

The script writes the browser dataset to `Dashboard/jobs.js`. Salary values
are stored as integer annual estimates with separate `salary_min` and
`salary_max` fields. Hourly ranges are annualized at 2,080 working hours.
Descriptions remain searchable in full and are visually shortened only on
the result cards.

## Open the dashboard

Serve the folder so the browser can load the generated data file:

```bash
cd Dashboard
python -m http.server 8000
```

Then open <http://127.0.0.1:8000/>.

## Deploy on Streamlit Cloud

Create a Streamlit app connected to the repository and set the main file to:

```text
capstone_project_2026/Dashboard/streamlit_app.py
```

If `capstone_project_2026` is already the selected repository root, use:

```text
Dashboard/streamlit_app.py
```

The Streamlit entry point embeds `index.html` and the generated `jobs.js`, so
this dashboard does not require API keys or PostgreSQL secrets.
