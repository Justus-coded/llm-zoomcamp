# Data Roles Assistant

Data Roles Assistant is a package and notebook-based retrieval-augmented generation (RAG) project for searching and answering questions about data-related job postings. It combines keyword/TF-IDF retrieval with OpenAI models and uses an LLM-as-a-judge evaluator to score generated answers.

The project is intended to help users explore roles such as:

- Business Analyst
- Data Analyst
- Data Engineer
- Data Scientist

## Problem statement

Job-posting datasets contain useful information about responsibilities, salary estimates, employers, locations, industries, and company details, but the information is difficult to search with simple keyword queries. This project provides a question-answering interface that retrieves relevant job postings and generates answers using only the retrieved context.

The notebook also evaluates retrieval quality using:

- Hit rate: whether the correct job posting appears in the top results.
- Mean reciprocal rank (MRR): how highly the correct job posting is ranked.

## Dataset

The main structured dataset is `data_jobs_pydantic.csv`. It contains validated job-posting records with these fields:

- `id`
- `job_title`
- `salary_estimate`
- `job_description`
- `rating`
- `company_name`
- `location`
- `headquarters`
- `size`
- `type_of_ownership`
- `industry`
- `sector`
- `competitors`
- `page_content`

`page_content` is a normalized text representation of the job fields and is used to create document embeddings for vector search.

The project also includes `data_jobs_questions.csv`, which contains generated job-search questions and the corresponding job IDs. These records are used as retrieval ground truth.

### Data limitations

- Some source rows were blank and were omitted during validation.
- The original rating field was serialized incorrectly as a pandas Series, so ratings are unavailable in the structured output.
- Salary estimates are stored as text ranges rather than numeric minimum and maximum values.
- The retrieval questions are generated evaluation data, not manually labeled questions.

## Project workflow

The main workflow is implemented in `Untitled.ipynb` and exported in `Untitled.py`. It:

1. Loads the structured job dataset into pandas.
2. Builds a basic `minsearch.Index` over job fields.
3. Applies field-specific boosts for titles, descriptions, locations, industries, and companies.
4. Creates document embeddings with `text-embedding-3-small`.
5. Builds a `minsearch.VectorSearch` index.
6. Combines basic and vector retrieval with reciprocal rank fusion (RRF).
7. Builds a grounded prompt from the retrieved job records.
8. Uses the OpenAI Responses API to generate an answer.
9. Evaluates every generated answer with an LLM-as-a-judge relevance classifier.
10. Stores conversations and user feedback in PostgreSQL.
11. Exposes the workflow through a Flask API.

## Package structure

The reusable application code is organized as follows:

```text
data_role_assistant/
├── __init__.py
├── ingest.py       # Load the CSV and build the minsearch index
└── rag.py          # Search, prompt construction, and LLM calls
app.py              # Flask API
db.py               # PostgreSQL persistence
streamlit_app.py    # Interactive RAG tester and monitoring dashboard
grafana/
├── dashboard.json   # SQL-backed Grafana dashboard
└── init.py          # Grafana datasource/dashboard provisioning
Dockerfile
docker-compose.yaml
```

`minsearch.py` is intentionally not copied into the project. The project uses the `minsearch` package installed through `uv`.

## Installation

Install [uv](https://docs.astral.sh/uv/) if it is not already installed, then run the following from the project directory:

```bash
uv sync
```

This creates or updates the project virtual environment and installs the dependencies declared in `pyproject.toml`.

## Environment variables

Copy `.env.example` to `.env` and set the credentials for your environment:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.6-luna
OPENAI_EVALUATION_MODEL=gpt-5.6-luna

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=data_role_assistant
POSTGRES_USER=user
POSTGRES_PASSWORD=password
```

The key is required for:

- Generating job questions.
- Creating document and query embeddings.
- Generating RAG answers.
- Evaluating generated answers with an LLM.

Never commit `.env` or expose the API key in a notebook output.

Each `/question` request makes one generation call and one evaluator call. The response includes the evaluator classification, token counts, estimated OpenAI cost, and response time.

The default model name is `gpt-5.6-luna` for both answer generation and evaluation. Replace it with a model available through your configured API endpoint if necessary.

## Run the application

The Flask API loads the search index on first use. By default it reads `data_jobs_pydantic.csv`. To use another CSV, set `DATA_PATH` before starting the server:

```bash
export DATA_PATH=data_jobs_pydantic.csv
uv run python app.py
```

The API listens on `http://127.0.0.1:5000` locally by default. Available endpoints:

```bash
curl http://127.0.0.1:5000/health
curl "http://127.0.0.1:5000/search?q=data%20analyst%20healthcare&num_results=5"
```

Questions are evaluated and saved to PostgreSQL with a `POST` request:

```bash
curl -X POST http://127.0.0.1:5000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "Find data analyst jobs in healthcare near New York"}'
```

Submit feedback using the returned conversation ID:

```bash
curl -X POST http://127.0.0.1:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "conversation-id-from-question-response", "feedback": 1}'
```

The `/search` endpoint uses the boosted basic retriever. The `/question` endpoint retrieves job postings, builds a grounded prompt, calls the configured OpenAI model and evaluator, and saves the result. `/rag` remains available as a backward-compatible alias.

## Run the Streamlit app

Streamlit provides a simple interface for testing questions and reviewing operational metrics:

```bash
uv run streamlit run streamlit_app.py
```

The app contains two views:

- **Ask**: submit a job-search question, view the grounded answer, LLM-as-a-judge relevance result, latency, token usage, estimated cost, and submit positive or negative feedback.
- **Dashboard**: view conversation count, average response time, estimated OpenAI cost, token usage, relevance classifications, feedback counts, and recent conversations.

The Ask view can generate an answer without PostgreSQL, but the conversation and feedback will not be saved. The Dashboard requires PostgreSQL and the initialized schema.

## PostgreSQL and Grafana

Start PostgreSQL, the Flask API, and Grafana with Docker Compose:

```bash
docker compose up -d
```

Initialize the database schema from the application container:

```bash
docker compose exec app python -c "import db; db.init_db()"
```

The default database contains:

- `conversations`: questions, answers, evaluation labels, token usage, estimated cost, and timestamps.
- `feedback`: positive (`1`) or negative (`-1`) feedback linked to a conversation.

Provision the PostgreSQL datasource and dashboard in Grafana:

```bash
uv run python grafana/init.py
```

The API is available at `http://localhost:5000` and Grafana is available at `http://localhost:3000`. The default Grafana login is `admin` / `admin`; change it through `GRAFANA_USER` and `GRAFANA_PASSWORD` before deployment.

`grafana/dashboard.json` contains SQL panels for conversation volume, response time, relevance classifications, OpenAI cost, and user feedback. `grafana/init.py` creates or updates the datasource and dashboard through the Grafana HTTP API.

## Run the notebook

The original exploratory and evaluation workflow is in `Untitled.ipynb`. Start Jupyter Lab with:

```bash
uv run jupyter lab
```

Then open `Untitled.ipynb` and run the cells in order. Later cells depend on indexes, prompts, API clients, embeddings, and evaluation data created by earlier cells.

After the setup cells have run, notebook queries can be tested with:

```python
question = "Find data analyst jobs in healthcare near New York"

basic_answer = rag1(question)
print(basic_answer)

hybrid_answer = hybrid_rag(question)
print(hybrid_answer)
```

`rag1` uses the boosted field-level basic search. `hybrid_rag` uses the hybrid search that combines basic retrieval and vector retrieval with RRF.

From the project directory:

```bash
uv run jupyter notebook Untitled.ipynb
```

Alternatively, use `uv run jupyter lab` and open the notebook from the browser interface.

`Untitled.html` is a static HTML export of a previous notebook execution. It is useful for reviewing saved outputs and does not execute the application.

`Untitled.py` is an exported notebook script. The notebook is the recommended execution path because the exported script contains notebook-state-dependent and optional cells.

## Retrieval evaluation

The notebook creates a validation/test split from `data_jobs_questions.csv`, searches for the expected job ID, and calculates hit rate and MRR. A typical evaluation call is:

```python
metrics = evaluate(
    gt_test,
    lambda q: hybrid_search(q["question"]),
)

print(metrics)
```

The repository also contains saved evaluation outputs:

- `rag1-eval-gpt-5.6-luna.csv`
- `hybrid-rag-eval-gpt-5.6-luna.csv`

### Retrieval benchmark

The project generated **2,546 evaluation questions** for the first **1,273 job rows**, with two questions generated per row.

Retrieval quality was measured with hit rate and mean reciprocal rank (MRR). Basic search without field boosting produced:

```text
{'hit_rate': 0.3157894736842105, 'mrr': 0.22479613212134808}
```

Random search was used to optimize field boosts over these ranges:

```python
param_ranges = {
    "job_title": (1.0, 6.0),
    "job_description": (0.5, 3.0),
    "location": (0.0, 3.0),
    "industry": (0.0, 2.5),
    "company_name": (0.0, 2.5),
    "sector": (0.0, 1.5),
    "salary_estimate": (0.0, 1.5),
}
```

The selected boost parameters were:

```python
boost = {
    "job_title": 3.63,
    "job_description": 1.61,
    "location": 0.75,
    "industry": 2.03,
    "company_name": 2.01,
    "sector": 0.01,
    "salary_estimate": 0.99,
}
```

After applying these boosts to the basic search, performance improved to:

```text
{'hit_rate': 0.5890052356020943, 'mrr': 0.41985442810050116}
```

Hybrid search using reciprocal rank fusion (RRF) improved performance further:

```text
{'hit_rate': 0.6387434554973822, 'mrr': 0.4699970913321694}
```

### LLM-as-a-judge evaluation

The project evaluated 200 randomly selected answers generated with hybrid RRF search. The relevance distribution was:

```text
RELEVANT           0.835
PARTLY_RELEVANT    0.110
NON_RELEVANT       0.055
```

## Generated artifacts

Running the notebook may create or update:

- `data_jobs_questions.csv`
- `job_vectors.npy`
- `rag1-eval-gpt-5.6-luna.csv`
- `hybrid-rag-eval-gpt-5.6-luna.csv`
- PostgreSQL rows in `conversations` and `feedback`

`job_vectors.npy` caches document embeddings so they do not need to be regenerated on every notebook run.
