#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from minsearch import Index

df = pd.read_csv("data_jobs_pydantic.csv")


# In[33]:


import time
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from minsearch import VectorSearch
from openai import OpenAI
from tqdm.auto import tqdm


# In[2]:


df.head()


# In[3]:


documents = df.to_dict(orient="records")

#documents


# In[4]:


df = df.set_index("id")


# In[5]:


index = Index(
    text_fields=["page_content"],
    keyword_fields=[
        "job_title",
        "company_name",
        "location",
        "industry",
        "sector",
        "type_of_ownership",
    ],
)
index.fit(documents)


# In[23]:


text_fields = [
    "job_title",
    "job_description",
    "company_name",
    "location",
    "industry",
    "sector",
    "salary_estimate",
]

keyword_fields = ["id",
    "location",
    "company_name",
    "industry",
    "sector",
    "type_of_ownership",
    "size",
]

index1 = Index(
    text_fields=text_fields,
    keyword_fields=keyword_fields,
)

index1.fit(df.to_dict(orient="records"))


# In[7]:


### Search Function

def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[8]:


### Search Function

def search1(query):
    boost = {}

    results = index1.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[9]:


prompt_template = """
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


entry_template = """
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


def get_value(document, field):
    value = document.get(field)
    if value is None or str(value).strip() in {"", "-1"}:
        return "Not listed"
    return str(value).strip()


def build_prompt(query, search_results):
    context_entries = []

    for document in search_results:
        context_entries.append(
            entry_template.format(
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
        )

    context = "\n\n---\n\n".join(context_entries)

    return prompt_template.format(
        question=query,
        context=context or "No matching job listings were found.",
    ).strip()


# In[10]:


import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path("/home/justus/llm_projects/data_roles_assistant/.env"))

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY was not loaded")

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm(prompt, model="gpt-5.6-luna"):
    response = openai_client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text


# In[11]:


def rag(query, model="gpt-5.6-luna"):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer


# In[12]:


def rag1(query, model="gpt-5.6-luna"):
    search_results = search1(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer


# In[13]:


question = "BA roles in AL"
answer = rag(question)
print(answer)


# In[14]:


question = "BA roles in AL"
answer = rag1(question)
print(answer)


# In[15]:


# ## Generating ground truth data

# import json
# import time
# from pathlib import Path

# import pandas as pd
# from dotenv import load_dotenv
# from openai import OpenAI
# from tqdm.auto import tqdm

# load_dotenv()

# openai_client = OpenAI()

# INPUT_FILE = Path("data_jobs_pydantic.csv")
# OUTPUT_FILE = Path("data_jobs_questions.csv")
# MODEL = "gpt-4o-mini"

# df = pd.read_csv(INPUT_FILE).fillna("")

# prompt1_template = """
# You are a data-career expert generating retrieval-evaluation questions.

# For the job listing below, generate exactly 2 distinct questions that a job seeker might ask.
# Questions must be answerable using only the information in the listing.
# Cover different aspects such as responsibilities, salary, location, company, industry, or qualifications.
# Do not invent information.

# Return only a valid JSON array containing objects with "id" and "question" fields.

# Job ID: {id}
# Job Title: {job_title}
# Salary Estimate: {salary_estimate}
# Company Name: {company_name}
# Location: {location}
# Headquarters: {headquarters}
# Company Size: {size}
# Type of Ownership: {type_of_ownership}
# Industry: {industry}
# Sector: {sector}
# Competitors: {competitors}

# Job Description:
# {job_description}
# """.strip()


# def parse_questions(output_text, job_id):
#     text = output_text.strip()

#     # Remove Markdown code fences if the model adds them
#     if text.startswith("```"):
#         text = text.replace("```json", "", 1)
#         text = text.removesuffix("```").strip()

#     questions = json.loads(text)

#     if not isinstance(questions, list) or len(questions) != 2:
#         raise ValueError("The model did not return exactly two questions")

#     results = []

#     for question in questions:
#         if not isinstance(question, dict) or not question.get("question"):
#             raise ValueError("Invalid question format")

#         results.append(
#             {
#                 "id": str(job_id),
#                 "question": question["question"].strip(),
#             }
#         )

#     return results


# def generate_questions(row, max_retries=5):
#     prompt = prompt1_template.format(**row.to_dict())

#     for attempt in range(max_retries):
#         try:
#             response = openai_client.responses.create(
#                 model=MODEL,
#                 input=[{"role": "user", "content": prompt}],
#             )

#             return parse_questions(response.output_text, row["id"])

#         except Exception:
#             if attempt == max_retries - 1:
#                 raise

#             time.sleep(2**attempt)


# results = []

# for _, row in tqdm(
#     df.iterrows(),
#     total=len(df),
#     desc="Generating job questions",
# ):
#     results.extend(generate_questions(row))

# df_questions = pd.DataFrame(results)
# df_questions.to_csv(OUTPUT_FILE, index=False)

# print(f"Saved {len(df_questions):,} questions to {OUTPUT_FILE}")


# In[16]:


df_questions = pd.DataFrame(results)
df_questions.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df_questions):,} questions to {OUTPUT_FILE}")


# In[17]:


##df_question = pd.read_csv("data/ground-truth-retrieval.csv")
ground_truth = df_questions.to_dict(orient="records")


# In[18]:


df_questions.tail()


# In[19]:


df.head()


# In[20]:


## Evaluating retrieval quality

def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        doc_id = q["id"]
        results = search_function(q)
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total)
    }


# In[21]:


evaluate(ground_truth, lambda q: search(q["question"]))


# In[24]:


evaluate(ground_truth, lambda q: search1(q["question"]))


# In[25]:


## Finding the best boost parameters

df_validation = df_questions[:1400]
df_test = df_questions[1400:]

gt_val = df_validation.to_dict(orient="records")
gt_test = df_test.to_dict(orient="records")


# In[26]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index1.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )
    return results


# In[27]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float("-inf")

    for _ in range(n_iterations):
        current_params = {}
        for field, (low, high) in param_ranges.items():
            current_params[field] = random.uniform(low, high)

        current_score = objective_function(current_params)

        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    return best_params

param_ranges = {
    "job_title": (1.0, 6.0),
    "job_description": (0.5, 3.0),
    "location": (0.0, 3.0),
    "industry": (0.0, 2.5),
    "company_name": (0.0, 2.5),
    "sector": (0.0, 1.5),
    "salary_estimate": (0.0, 1.5),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q["question"], boost=boost_params)

    results = evaluate(gt_val, search_function)
    return results["hit_rate"]

best_params = simple_optimize(param_ranges, objective, n_iterations=20)
print("Best boost parameters:", best_params)


# In[28]:


### Updated 
def search_b(query):
    boost ={
    "job_title": 3.63,
    "job_description": 1.61,
    "location": 0.75,
    "industry": 2.03,
    "company_name": 2.01,
    "sector": 0.01,
    "salary_estimate": 0.99,
}

    results = index1.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results



evaluate(gt_test, lambda q: search_b(q["question"]))


# In[ ]:


## previous runnnn 

def search_b(query):
    boost ={
    "job_title": 2.63,
    "job_description": 2.92,
    "location": 1.49,
    "industry": 0.64,
    "company_name": 1.64,
    "sector": 0.01,
    "salary_estimate": 0.98,
}

    results = index1.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[51]:


evaluate(gt_test, lambda q: search_b(q["question"]))


# In[30]:


import numpy as np


# In[38]:


boost ={
    "job_title": 3.63,
    "job_description": 1.61,
    "location": 0.75,
    "industry": 2.03,
    "company_name": 2.01,
    "sector": 0.01,
    "salary_estimate": 0.99,
}


# In[32]:


### create and cache document embedding

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_FILE = Path("job_vectors.npy")


def embed_documents(texts, batch_size=64):
    vectors = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding documents",
    ):
        batch = texts[start:start + batch_size]

        for attempt in range(5):
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )

                batch_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                vectors.extend(item.embedding for item in batch_data)
                break

            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    return np.asarray(vectors, dtype=np.float32)


document_texts = [
    str(document.get("page_content", ""))[:24000]
    for document in documents
]

if VECTOR_FILE.exists():
    vectors = np.load(VECTOR_FILE)

    if len(vectors) != len(documents):
        vectors = embed_documents(document_texts)
        np.save(VECTOR_FILE, vectors)
else:
    vectors = embed_documents(document_texts)
    np.save(VECTOR_FILE, vectors)


# In[39]:


vector_index = VectorSearch(
    keyword_fields=[
        "location",
        "company_name",
        "industry",
        "sector",
        "type_of_ownership",
        "size",
    ]
)

vector_index.fit(vectors, documents)


# In[40]:


def embed_query(query):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )

    return np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    )


def hybrid_search(
    query,
    filter_dict=None,
    num_results=10,
    candidate_count=50,
    basic_weight=1.0,
    vector_weight=1.0,
):
    if filter_dict is None:
        filter_dict = {}

    basic_results = index1.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost,
        num_results=candidate_count,
    )

    query_vector = embed_query(query)

    vector_results = vector_index.search(
        query_vector=query_vector,
        filter_dict=filter_dict,
        num_results=candidate_count,
    )

    scores = {}
    ranked_documents = {}

    # Reciprocal Rank Fusion
    rank_constant = 60

    for rank, document in enumerate(basic_results, start=1):
        document_id = str(document["id"])
        ranked_documents[document_id] = document
        scores[document_id] = scores.get(document_id, 0) + (
            basic_weight / (rank_constant + rank)
        )

    for rank, document in enumerate(vector_results, start=1):
        document_id = str(document["id"])
        ranked_documents[document_id] = document
        scores[document_id] = scores.get(document_id, 0) + (
            vector_weight / (rank_constant + rank)
        )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    results = []

    for document_id in ranked_ids[:num_results]:
        document = dict(ranked_documents[document_id])
        document["_hybrid_score"] = scores[document_id]
        results.append(document)

    return results


# In[41]:


results = hybrid_search(
    "Find data analyst jobs in healthcare near New York",
    num_results=10,
)

for result in results[:3]:
    print(result["job_title"], result["company_name"])


# In[42]:


for result in results[:3]:
    print(result)


# In[43]:


hybrid_metrics = evaluate(
    gt_test,
    lambda q: hybrid_search(q["question"]),
)
print(hybrid_metrics)


# In[44]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as 'NON_RELEVANT', 'PARTLY_RELEVANT', or 'RELEVANT'.

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  'Relevance': 'NON_RELEVANT' | 'PARTLY_RELEVANT' | 'RELEVANT',
  'Explanation': '[Provide a brief explanation for your evaluation]'
}}
""".strip()


# In[46]:


import json
from tqdm.auto import tqdm

df_sample = df_questions.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient="records")

evaluations = []

for record in tqdm(sample):
    question = record["question"]
    answer_llm = rag1(question)

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[47]:


def hybrid_rag(query, model="gpt-5.6-luna"):
    search_results = hybrid_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer


# In[48]:


df_eval = pd.DataFrame(evaluations, columns=["record", "answer", "evaluation"])

df_eval["id"] = df_eval.record.apply(lambda d: d["id"])
df_eval["question"] = df_eval.record.apply(lambda d: d["question"])
df_eval["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
df_eval["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])

df_eval.relevance.value_counts(normalize=True)


# In[52]:


df_eval.to_csv("rag1-eval-gpt-5.6-luna.csv", index=False)


# In[49]:


import json
from tqdm.auto import tqdm

df_sample = df_questions.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient="records")

evaluations_hybrid = []

for record in tqdm(sample):
    question = record["question"]
    answer_llm = hybrid_rag(question)

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations_hybrid.append((record, answer_llm, evaluation))


# In[51]:


df_eval1 = pd.DataFrame(evaluations_hybrid, columns=["record", "answer", "evaluation"])

df_eval1["id"] = df_eval.record.apply(lambda d: d["id"])
df_eval1["question"] = df_eval.record.apply(lambda d: d["question"])
df_eval1["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
df_eval1["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])

df_eval1.relevance.value_counts(normalize=True)


# In[53]:


df_eval1.to_csv("hybrid-rag-eval-gpt-5.6-luna.csv", index=False)


# In[ ]:





# In[54]:


jupyter nbconvert --to=script Untitled.ipynb


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


#uv add langchain langchain-elasticsearch langchain-huggingface sentence-transformers


# In[57]:


from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

es_url = "http://localhost:9200"


# In[ ]:




