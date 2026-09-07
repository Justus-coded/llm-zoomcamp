```python
import pandas as pd
from minsearch import Index

df = pd.read_csv("data_jobs_pydantic.csv")

```


```python
import time
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from minsearch import VectorSearch
from openai import OpenAI
from tqdm.auto import tqdm
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job_title</th>
      <th>salary_estimate</th>
      <th>job_description</th>
      <th>rating</th>
      <th>company_name</th>
      <th>location</th>
      <th>headquarters</th>
      <th>size</th>
      <th>type_of_ownership</th>
      <th>industry</th>
      <th>sector</th>
      <th>competitors</th>
      <th>page_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>job-00001</td>
      <td>Business Analyst - Clinical &amp; Logistics Platform</td>
      <td>$56K-$102K</td>
      <td>Company Overview\n\n\nAt Memorial Sloan Ketter...</td>
      <td>NaN</td>
      <td>Memorial Sloan-Kettering</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>10000+ employees</td>
      <td>Nonprofit Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>Mayo Clinic, The Johns Hopkins Hospital, MD An...</td>
      <td>Job Title: Business Analyst - Clinical &amp; Logis...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>job-00002</td>
      <td>Business Analyst</td>
      <td>$56K-$102K</td>
      <td>We are seeking for an energetic and collaborat...</td>
      <td>NaN</td>
      <td>Paine Schwartz Partners</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1 to 50 employees</td>
      <td>Company - Private</td>
      <td>Venture Capital &amp; Private Equity</td>
      <td>Finance</td>
      <td>-1</td>
      <td>Job Title: Business Analyst\nSalary Estimate: ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>job-00003</td>
      <td>Data Analyst</td>
      <td>$56K-$102K</td>
      <td>For more than a decade, Asembia has been worki...</td>
      <td>NaN</td>
      <td>Asembia</td>
      <td>Florham Park, NJ</td>
      <td>Florham Park, NJ</td>
      <td>501 to 1000 employees</td>
      <td>Company - Private</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>-1</td>
      <td>Job Title: Data Analyst\nSalary Estimate: $56K...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>job-00004</td>
      <td>Information Security Analyst, Incident Response</td>
      <td>$56K-$102K</td>
      <td>Job Description Summary\nThe Information Secur...</td>
      <td>NaN</td>
      <td>BD</td>
      <td>Franklin Lakes, NJ</td>
      <td>Franklin Lakes, NJ</td>
      <td>10000+ employees</td>
      <td>Company - Public</td>
      <td>Health Care Products Manufacturing</td>
      <td>Manufacturing</td>
      <td>Abbott, Siemens, Baxter</td>
      <td>Job Title: Information Security Analyst, Incid...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>job-00005</td>
      <td>Analyst - FP&amp;A Global Revenue</td>
      <td>$56K-$102K</td>
      <td>Magnite is the world's largest independent sel...</td>
      <td>NaN</td>
      <td>Rubicon Project</td>
      <td>New York, NY</td>
      <td>Los Angeles, CA</td>
      <td>201 to 500 employees</td>
      <td>Company - Public</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>PubMatic, AppNexus, Index Exchange</td>
      <td>Job Title: Analyst - FP&amp;A Global Revenue\nSala...</td>
    </tr>
  </tbody>
</table>
</div>




```python
documents = df.to_dict(orient="records")

#documents
```


```python
df = df.set_index("id")
```


```python
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
```




    <minsearch.minsearch.Index at 0x7f55fe8b0b50>




```python
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
```




    <minsearch.minsearch.Index at 0x7f55f55f3990>




```python
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
```


```python
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
```


```python
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
```


```python
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
```


```python
def rag(query, model="gpt-5.6-luna"):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer
```


```python
def rag1(query, model="gpt-5.6-luna"):
    search_results = search1(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer
```


```python
question = "BA roles in AL"
answer = rag(question)
print(answer)
```

    No direct **Business Analyst (BA) roles in Alabama** are listed.
    
    The closest Alabama-based listing is:
    
    - **Data Engineer 4 – Guidewire Data Hub — Kemper**
      - **Location:** Birmingham, AL or Jacksonville, FL
      - **Salary estimate:** $61K–$94K
      - **Reason:** The role involves requirements, analysis, and the development lifecycle, but it is primarily an ETL/BI data engineering position—not a BA role.
    
    A duplicate listing for the same Kemper position shows a salary estimate of **$27K–$50K**.



```python
question = "BA roles in AL"
answer = rag1(question)
print(answer)
```

    No BA roles in Alabama (AL) are listed in the provided job database. All listed roles are located in Indianapolis, Indiana, including:
    
    - **Business Data Analyst** — The Fountain Group — Indianapolis, IN — **$34K–$61K** — Focuses on business process analysis, requirements gathering, data validation, dashboards, and reporting.
    - **Data Analyst (BA)** — Briljent — Indianapolis, IN — **$34K–$61K** — Supports requirements gathering, project documentation, risk analysis, metrics, and stakeholder reporting.
    
    The database does not list any Alabama locations.



```python

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
```


    Generating job questions:   0%|          | 0/12381 [00:00<?, ?it/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[15], line 104
        100     df.iterrows(),
        101     total=len(df),
        102     desc="Generating job questions",
        103 ):
    --> 104     results.extend(generate_questions(row))
        105 
        106 df_questions = pd.DataFrame(results)
        107 df_questions.to_csv(OUTPUT_FILE, index=False)


    Cell In[15], line 90, in generate_questions(row, max_retries)
         86             )
         87 
         88             return parse_questions(response.output_text, row["id"])
         89 
    ---> 90         except Exception:
         91             if attempt == max_retries - 1:
         92                 raise
         93 


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/resources/responses/responses.py:1023, in Responses.create(self, background, context_management, conversation, include, input, instructions, max_output_tokens, max_tool_calls, metadata, model, moderation, parallel_tool_calls, previous_response_id, prompt, prompt_cache_key, prompt_cache_options, prompt_cache_retention, reasoning, safety_identifier, service_tier, store, stream, stream_options, temperature, text, tool_choice, tools, top_logprobs, top_p, truncation, user, extra_headers, extra_query, extra_body, timeout)
        982 def create(
        983     self,
        984     *,
       (...)   1021     timeout: float | httpx2.Timeout | None | NotGiven = not_given,
       1022 ) -> Response | Stream[ResponseStreamEvent]:
    -> 1023     return self._post(
       1024         "/responses",
       1025         body=maybe_transform(
       1026             {
       1027                 "background": background,
       1028                 "context_management": context_management,
       1029                 "conversation": conversation,
       1030                 "include": include,
       1031                 "input": input,
       1032                 "instructions": instructions,
       1033                 "max_output_tokens": max_output_tokens,
       1034                 "max_tool_calls": max_tool_calls,
       1035                 "metadata": metadata,
       1036                 "model": model,
       1037                 "moderation": moderation,
       1038                 "parallel_tool_calls": parallel_tool_calls,
       1039                 "previous_response_id": previous_response_id,
       1040                 "prompt": prompt,
       1041                 "prompt_cache_key": prompt_cache_key,
       1042                 "prompt_cache_options": prompt_cache_options,
       1043                 "prompt_cache_retention": prompt_cache_retention,
       1044                 "reasoning": reasoning,
       1045                 "safety_identifier": safety_identifier,
       1046                 "service_tier": service_tier,
       1047                 "store": store,
       1048                 "stream": stream,
       1049                 "stream_options": stream_options,
       1050                 "temperature": temperature,
       1051                 "text": text,
       1052                 "tool_choice": tool_choice,
       1053                 "tools": tools,
       1054                 "top_logprobs": top_logprobs,
       1055                 "top_p": top_p,
       1056                 "truncation": truncation,
       1057                 "user": user,
       1058             },
       1059             response_create_params.ResponseCreateParamsStreaming
       1060             if stream
       1061             else response_create_params.ResponseCreateParamsNonStreaming,
       1062         ),
       1063         options=make_request_options(
       1064             extra_headers=extra_headers,
       1065             extra_query=extra_query,
       1066             extra_body=extra_body,
       1067             timeout=timeout,
       1068             security={"bearer_auth": True},
       1069         ),
       1070         cast_to=Response,
       1071         stream=stream or False,
       1072         stream_cls=Stream[ResponseStreamEvent],
       1073     )


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/_base_client.py:1368, in SyncAPIClient.post(self, path, cast_to, body, content, options, files, stream, stream_cls)
       1359     warnings.warn(
       1360         "Passing raw bytes as `body` is deprecated and will be removed in a future version. "
       1361         "Please pass raw bytes via the `content` parameter instead.",
       1362         DeprecationWarning,
       1363         stacklevel=2,
       1364     )
       1365 opts = FinalRequestOptions.construct(
       1366     method="post", url=path, json_data=body, content=content, files=to_httpx_files(files), **options
       1367 )
    -> 1368 return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/_base_client.py:1076, in SyncAPIClient.request(self, cast_to, options, stream, stream_cls)
       1074 response = None
       1075 try:
    -> 1076     response = self._send_request(
       1077         request,
       1078         stream=stream or self._should_stream_response_body(request=request),
       1079         **kwargs,
       1080     )
       1081 except timeout_exceptions() as err:
       1082     log.debug("Encountered a timeout exception: %s", type(err).__name__)


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/_client.py:591, in OpenAI._send_request(self, request, stream, **kwargs)
        583 @override
        584 def _send_request(
        585     self,
       (...)    589     **kwargs: Unpack[HttpxSendArgs],
        590 ) -> httpx2.Response:
    --> 591     response = self._send_with_auth_retry(request, stream=stream, **kwargs)
        592     if self._provider_runtime is not None and self._provider_runtime.normalize_response is not None:
        593         response = self._provider_runtime.normalize_response(response)


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/_client.py:570, in OpenAI._send_with_auth_retry(self, request, stream, retried, **kwargs)
        562     response = x509_auth.send_api_request(
        563         request,
        564         expected_origin=self.base_url,
       (...)    567         **kwargs,
        568     )
        569 else:
    --> 570     response = super()._send_request(request, stream=stream, **kwargs)
        571 if response.status_code != 401 or self._workload_identity_auth is None or used_access_token is None:
        572     return response


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/openai/_base_client.py:999, in SyncAPIClient._send_request(self, request, stream, **kwargs)
        992 def _send_request(
        993     self,
        994     request: httpx2.Request,
       (...)    997     **kwargs: Unpack[HttpxSendArgs],
        998 ) -> httpx2.Response:
    --> 999     return self._client.send(request, stream=stream, **kwargs)


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpx2/_client.py:980, in Client.send(self, request, stream, auth, follow_redirects)
        976 self._set_timeout(request)
        978 auth = self._build_request_auth(request, auth)
    --> 980 response = self._send_handling_auth(
        981     request,
        982     auth=auth,
        983     follow_redirects=follow_redirects,
        984     history=[],
        985 )
        986 try:
        987     if not stream:


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpx2/_client.py:1008, in Client._send_handling_auth(self, request, auth, follow_redirects, history)
       1005 request = next(auth_flow)
       1007 while True:
    -> 1008     response = self._send_handling_redirects(
       1009         request,
       1010         follow_redirects=follow_redirects,
       1011         history=history,
       1012     )
       1013     try:
       1014         try:


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpx2/_client.py:1043, in Client._send_handling_redirects(self, request, follow_redirects, history)
       1040 for hook in self._event_hooks["request"]:
       1041     hook(request)
    -> 1043 response = self._send_single_request(request)
       1044 try:
       1045     for hook in self._event_hooks["response"]:


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpx2/_client.py:1076, in Client._send_single_request(self, request)
       1073     raise RuntimeError("Attempted to send an async request with a sync Client instance.")
       1075 with request_context(request=request):
    -> 1076     response = transport.handle_request(request)
       1078 assert isinstance(response.stream, SyncByteStream)
       1080 response.request = request


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpx2/_transports/default.py:245, in HTTPTransport.handle_request(self, request)
        232 req = httpcore2.Request(
        233     method=request.method,
        234     url=httpcore2.URL(
       (...)    242     extensions=request.extensions,
        243 )
        244 with map_httpcore_exceptions():
    --> 245     resp = self._pool.handle_request(req)
        247 assert isinstance(resp.stream, typing.Iterable)
        249 return Response(
        250     status_code=resp.status,
        251     headers=resp.headers,
        252     stream=ResponseStream(resp.stream),
        253     extensions=resp.extensions,
        254 )


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/connection_pool.py:242, in ConnectionPool.handle_request(self, request)
        239         closing = self._assign_requests_to_connections()
        241     self._close_connections(closing)
    --> 242     raise exc from None
        244 # Return the response. Note that in this case we still have to manage
        245 # the point at which the response is closed.
        246 assert isinstance(response.stream, typing.Iterable)


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/connection_pool.py:224, in ConnectionPool.handle_request(self, request)
        220 connection = pool_request.wait_for_connection(timeout=timeout)
        222 try:
        223     # Send the request on the assigned connection.
    --> 224     response = connection.handle_request(pool_request.request)
        225 except ConnectionNotAvailable:
        226     # In some cases a connection may initially be available to
        227     # handle a request, but then become unavailable.
        228     #
        229     # In this case we clear the connection and try again.
        230     pool_request.clear_connection()


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/connection.py:96, in HTTPConnection.handle_request(self, request)
         93     self._connect_failed = True
         94     raise exc
    ---> 96 return self._connection.handle_request(request)


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/http11.py:125, in HTTP11Connection.handle_request(self, request)
        123     with Trace("response_closed", logger, request) as trace:
        124         self._response_closed()
    --> 125 raise exc


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/http11.py:97, in HTTP11Connection.handle_request(self, request)
         88     pass
         90 with Trace("receive_response_headers", logger, request, kwargs) as trace:
         91     (
         92         http_version,
         93         status,
         94         reason_phrase,
         95         headers,
         96         trailing_data,
    ---> 97     ) = self._receive_response_headers(**kwargs)
         98     trace.return_value = (
         99         http_version,
        100         status,
        101         reason_phrase,
        102         headers,
        103     )
        105 network_stream = self._network_stream


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/http11.py:167, in HTTP11Connection._receive_response_headers(self, request)
        164 timeout = timeouts.get("read", None)
        166 while True:
    --> 167     event = self._receive_event(timeout=timeout)
        168     if isinstance(event, h11.Response):
        169         break


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_sync/http11.py:200, in HTTP11Connection._receive_event(self, timeout)
        197     event = self._h11_state.next_event()
        199 if event is h11.NEED_DATA:
    --> 200     data = self._network_stream.read(self.READ_NUM_BYTES, timeout=timeout)
        202     # If we feed this case through h11 we'll raise an exception like:
        203     #
        204     #     httpcore2.RemoteProtocolError: can't handle event type
       (...)    208     # perspective. Instead we handle this case distinctly and treat
        209     # it as a ConnectError.
        210     if data == b"" and self._h11_state.their_state == h11.SEND_RESPONSE:


    File ~/llm_projects/data_roles_assistant/.venv/lib/python3.11/site-packages/httpcore2/_backends/sync.py:127, in SyncStream.read(self, max_bytes, timeout)
        125 with map_exceptions(exc_map):
        126     self._sock.settimeout(timeout)
    --> 127     return self._sock.recv(max_bytes)


    File ~/anaconda3/lib/python3.11/ssl.py:1296, in SSLSocket.recv(self, buflen, flags)
       1292     if flags != 0:
       1293         raise ValueError(
       1294             "non-zero flags not allowed in calls to recv() on %s" %
       1295             self.__class__)
    -> 1296     return self.read(buflen)
       1297 else:
       1298     return super().recv(buflen, flags)


    File ~/anaconda3/lib/python3.11/ssl.py:1169, in SSLSocket.read(self, len, buffer)
       1167         return self._sslobj.read(len, buffer)
       1168     else:
    -> 1169         return self._sslobj.read(len)
       1170 except SSLError as x:
       1171     if x.args[0] == SSL_ERROR_EOF and self.suppress_ragged_eofs:


    KeyboardInterrupt: 



```python

df_questions = pd.DataFrame(results)
df_questions.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df_questions):,} questions to {OUTPUT_FILE}")
```

    Saved 2,546 questions to data_jobs_questions.csv



```python
##df_question = pd.read_csv("data/ground-truth-retrieval.csv")
ground_truth = df_questions.to_dict(orient="records")
```


```python
df_questions.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>question</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2541</th>
      <td>job-01271</td>
      <td>What kind of experience is required for this r...</td>
    </tr>
    <tr>
      <th>2542</th>
      <td>job-01272</td>
      <td>What is the salary range for the MDM Data Anal...</td>
    </tr>
    <tr>
      <th>2543</th>
      <td>job-01272</td>
      <td>What qualifications are required for the MDM D...</td>
    </tr>
    <tr>
      <th>2544</th>
      <td>job-01273</td>
      <td>What is the salary estimate for the Business C...</td>
    </tr>
    <tr>
      <th>2545</th>
      <td>job-01273</td>
      <td>Where is the location for the Business Consult...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job_title</th>
      <th>salary_estimate</th>
      <th>job_description</th>
      <th>rating</th>
      <th>company_name</th>
      <th>location</th>
      <th>headquarters</th>
      <th>size</th>
      <th>type_of_ownership</th>
      <th>industry</th>
      <th>sector</th>
      <th>competitors</th>
      <th>page_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>job-00001</td>
      <td>Business Analyst - Clinical &amp; Logistics Platform</td>
      <td>$56K-$102K</td>
      <td>Company Overview\n\n\nAt Memorial Sloan Ketter...</td>
      <td></td>
      <td>Memorial Sloan-Kettering</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>10000+ employees</td>
      <td>Nonprofit Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>Mayo Clinic, The Johns Hopkins Hospital, MD An...</td>
      <td>Job Title: Business Analyst - Clinical &amp; Logis...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>job-00002</td>
      <td>Business Analyst</td>
      <td>$56K-$102K</td>
      <td>We are seeking for an energetic and collaborat...</td>
      <td></td>
      <td>Paine Schwartz Partners</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1 to 50 employees</td>
      <td>Company - Private</td>
      <td>Venture Capital &amp; Private Equity</td>
      <td>Finance</td>
      <td>-1</td>
      <td>Job Title: Business Analyst\nSalary Estimate: ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>job-00003</td>
      <td>Data Analyst</td>
      <td>$56K-$102K</td>
      <td>For more than a decade, Asembia has been worki...</td>
      <td></td>
      <td>Asembia</td>
      <td>Florham Park, NJ</td>
      <td>Florham Park, NJ</td>
      <td>501 to 1000 employees</td>
      <td>Company - Private</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>-1</td>
      <td>Job Title: Data Analyst\nSalary Estimate: $56K...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>job-00004</td>
      <td>Information Security Analyst, Incident Response</td>
      <td>$56K-$102K</td>
      <td>Job Description Summary\nThe Information Secur...</td>
      <td></td>
      <td>BD</td>
      <td>Franklin Lakes, NJ</td>
      <td>Franklin Lakes, NJ</td>
      <td>10000+ employees</td>
      <td>Company - Public</td>
      <td>Health Care Products Manufacturing</td>
      <td>Manufacturing</td>
      <td>Abbott, Siemens, Baxter</td>
      <td>Job Title: Information Security Analyst, Incid...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>job-00005</td>
      <td>Analyst - FP&amp;A Global Revenue</td>
      <td>$56K-$102K</td>
      <td>Magnite is the world's largest independent sel...</td>
      <td></td>
      <td>Rubicon Project</td>
      <td>New York, NY</td>
      <td>Los Angeles, CA</td>
      <td>201 to 500 employees</td>
      <td>Company - Public</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>PubMatic, AppNexus, Index Exchange</td>
      <td>Job Title: Analyst - FP&amp;A Global Revenue\nSala...</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


```python
evaluate(ground_truth, lambda q: search(q["question"]))
```


      0%|          | 0/2546 [00:00<?, ?it/s]





    {'hit_rate': 0.4964650432050275, 'mrr': 0.33264192820359406}




```python
evaluate(ground_truth, lambda q: search1(q["question"]))
```


      0%|          | 0/2546 [00:00<?, ?it/s]





    {'hit_rate': 0.3157894736842105, 'mrr': 0.22479613212134808}




```python
## Finding the best boost parameters

df_validation = df_questions[:1400]
df_test = df_questions[1400:]

gt_val = df_validation.to_dict(orient="records")
gt_test = df_test.to_dict(orient="records")
```


```python
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
```


```python
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
```


      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]



      0%|          | 0/1400 [00:00<?, ?it/s]


    Best boost parameters: {'job_title': 3.6366931201689, 'job_description': 1.6062297452484087, 'location': 0.7535953498234211, 'industry': 2.028455316307554, 'company_name': 2.009030600433723, 'sector': 0.012637824476107007, 'salary_estimate': 0.9939067397891572}



```python
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
```


      0%|          | 0/1146 [00:00<?, ?it/s]





    {'hit_rate': 0.5890052356020943, 'mrr': 0.41985442810050116}




```python
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
```


```python
evaluate(gt_test, lambda q: search_b(q["question"]))
```


      0%|          | 0/1070 [00:00<?, ?it/s]





    {'hit_rate': 0.585981308411215, 'mrr': 0.41888703456460413}




```python
import numpy as np
```


```python

boost ={
    "job_title": 3.63,
    "job_description": 1.61,
    "location": 0.75,
    "industry": 2.03,
    "company_name": 2.01,
    "sector": 0.01,
    "salary_estimate": 0.99,
}
```


```python
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
```


    Embedding documents:   0%|          | 0/194 [00:00<?, ?it/s]



```python
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
```




    <minsearch.vector.VectorSearch at 0x7f560df68c50>




```python
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
```


```python
results = hybrid_search(
    "Find data analyst jobs in healthcare near New York",
    num_results=10,
)

for result in results[:3]:
    print(result["job_title"], result["company_name"])
```

    Healthcare Data Analyst Fair Health
    NY Healthcare Data/Reporting Analyst SAT Healthcare
    Senior Healthcare Data Analyst Cecelia Health



```python
for result in results[:3]:
    print(result)
```

    {'id': 'job-04192', 'job_title': 'Healthcare Data Analyst', 'salary_estimate': '$51K-$87K', 'job_description': 'Healthcare Data Analyst\n\nPosition Description\n\n\nThe Healthcare Data Analyst will work on multiple aspects of data processing and analytics at FAIR Health, including but not limited to the processes used to create FAIR Health data products and the analytics used to support and evaluate those products. Working with multiple FAIR Health stakeholders, this position will have responsibilities throughout the product design and development lifecycle, translating business needs into data driven deliverables and analyses.\n\nThis person will have a strong analytic background and have an ability to read and understand data and extract information from it. They will be able to develop reports and analyses which clearly identify and communicate results. This person will also have an understanding of healthcare claims data and a working knowledge of coding (CPT, HCPCS, ICD, etc.).\n\nPrimary Responsibilities\nBuild and automate custom data processes using workflow tools.\nTranslate business needs into data driven deliverables and analyses.\nDesign and develop analytic reports and data visualizations.\nUnderstand and analyze healthcare claims data to identify trends, patterns and other salient information.\nCreate, merge, and query large datasets.\nInterpret data analysis results and present conclusions to multiple internal teams.\nWork with statistical analytical toolsets such as SAS/R to produce database tables, statistical results and graphical representations.\nSkills and Requirements\nRequired\nStrong analytical skills\nAbility to conduct research and analysis on large data sets, interpret results, and communicate findings.\nProficient in SQL, PL/SQL, or equivalent database query language.\nFunctional knowledge of workflow engines and batch jobs.\nExperience working with large databases.\nStrong knowledge of business intelligence tools, such as Tableau, SAS, etc.\nExpert ability with Excel.\nStrong interpersonal skills, including the ability to work in a highly cross-functional, multi-location organization.\nBS in Statistics, Mathematics, Computer Science, Public Health, or related degree.\nPreferred\nExperience working as a data analyst.\nExperience working in the healthcare industry.\nUnderstanding of healthcare data and analyses, including but not limited to medical claims data.\nWorking knowledge of healthcare coding (CPT, HCPCS, ICD, revenue codes, DRG).\nMS in Statistics, Mathematics, Computer Science, Public Health, or related degree.\nInterested candidates should submit their resume to resumes@fairhealth.org. Please include “Healthcare Data Analyst” and your last name in the subject line.\n\nFAIR Health, Inc. is an equal opportunity employer and an e-verify participant.', 'rating': nan, 'company_name': 'Fair Health', 'location': 'New York, NY', 'headquarters': 'New York, NY', 'size': '1 to 50 employees', 'type_of_ownership': 'Nonprofit Organization', 'industry': '-1', 'sector': '-1', 'competitors': '-1', 'page_content': 'Job Title: Healthcare Data Analyst\nSalary Estimate: $51K-$87K\nJob Description: Healthcare Data Analyst\n\nPosition Description\n\n\nThe Healthcare Data Analyst will work on multiple aspects of data processing and analytics at FAIR Health, including but not limited to the processes used to create FAIR Health data products and the analytics used to support and evaluate those products. Working with multiple FAIR Health stakeholders, this position will have responsibilities throughout the product design and development lifecycle, translating business needs into data driven deliverables and analyses.\n\nThis person will have a strong analytic background and have an ability to read and understand data and extract information from it. They will be able to develop reports and analyses which clearly identify and communicate results. This person will also have an understanding of healthcare claims data and a working knowledge of coding (CPT, HCPCS, ICD, etc.).\n\nPrimary Responsibilities\nBuild and automate custom data processes using workflow tools.\nTranslate business needs into data driven deliverables and analyses.\nDesign and develop analytic reports and data visualizations.\nUnderstand and analyze healthcare claims data to identify trends, patterns and other salient information.\nCreate, merge, and query large datasets.\nInterpret data analysis results and present conclusions to multiple internal teams.\nWork with statistical analytical toolsets such as SAS/R to produce database tables, statistical results and graphical representations.\nSkills and Requirements\nRequired\nStrong analytical skills\nAbility to conduct research and analysis on large data sets, interpret results, and communicate findings.\nProficient in SQL, PL/SQL, or equivalent database query language.\nFunctional knowledge of workflow engines and batch jobs.\nExperience working with large databases.\nStrong knowledge of business intelligence tools, such as Tableau, SAS, etc.\nExpert ability with Excel.\nStrong interpersonal skills, including the ability to work in a highly cross-functional, multi-location organization.\nBS in Statistics, Mathematics, Computer Science, Public Health, or related degree.\nPreferred\nExperience working as a data analyst.\nExperience working in the healthcare industry.\nUnderstanding of healthcare data and analyses, including but not limited to medical claims data.\nWorking knowledge of healthcare coding (CPT, HCPCS, ICD, revenue codes, DRG).\nMS in Statistics, Mathematics, Computer Science, Public Health, or related degree.\nInterested candidates should submit their resume to resumes@fairhealth.org. Please include “Healthcare Data Analyst” and your last name in the subject line.\n\nFAIR Health, Inc. is an equal opportunity employer and an e-verify participant.\nRating: Not available\nCompany Name: Fair Health\nLocation: New York, NY\nHeadquarters: New York, NY\nSize: 1 to 50 employees\nType of ownership: Nonprofit Organization\nIndustry: -1\nSector: -1\nCompetitors: -1', '_hybrid_score': 0.03149801587301587}
    {'id': 'job-04468', 'job_title': 'NY Healthcare Data/Reporting Analyst', 'salary_estimate': '$98K-$114K', 'job_description': 'Position Summary\nA data management/analyst who enjoys working with healthcare data so as to support clinicians, software developers, and senior leadership with the information and self help tools they need to succeed in growing our business. A lively curiosity about health care data and analytical tools is greatly desired. Position will work remote for now, applicants should live in the New York City area.\nJob Responsibilities:\nUnderstand the needs and opportunities across our internal and external stakeholders\nCreate, update, and distribute analytic reports on a recurring basis\nLearn new data related skills and processes\nLearn health care claim (UB04, HCFA 1500) formats and analysis\nShare and teach skills to others on the data team\nWork as part of a larger team and helping other teams in achieving their goals\nContinue to push the boundaries of what technology can do to empower our caregivers and clinicians to improve health outcomes for our patients\nRequirements:\nA College degree (required)\n5+ years experience working with health care data (e.g. claims, app output, CMS HCC, SDOH, clinical research) in queries or BI tools (required)\nExperience analyzing operational data and correlations it may have to clinical outcomes data\nMS Excel or Google Sheets advanced ability (pivot tables, formatting, graphics) (required)\nExperience with BI tools such as Tableau (highly desired)\nExperience with SQL, SAS, Python, SPSS or other analytic tools for data management and informatics\nAttention to detail along with a true love of data patterns, error detection and correction\nAbility to be a story-teller with data - to see and articulate clearly the story behind the numbers\nIntellectual curiosity around finding the why and digging deep to find patterns others might miss\nPassion about our mission to improve people s lives\nComfort in a dynamic and always evolving start-up environment\n\nMinimum Years of Experience: 5\n\nSub Specialties: Business Analyst- Associate', 'rating': nan, 'company_name': 'SAT Healthcare', 'location': 'New York, NY', 'headquarters': 'Santa Clara, CA', 'size': '201 to 500 employees', 'type_of_ownership': 'Company - Private', 'industry': 'Health Care Services & Hospitals', 'sector': 'Health Care', 'competitors': '-1', 'page_content': 'Job Title: NY Healthcare Data/Reporting Analyst\nSalary Estimate: $98K-$114K\nJob Description: Position Summary\nA data management/analyst who enjoys working with healthcare data so as to support clinicians, software developers, and senior leadership with the information and self help tools they need to succeed in growing our business. A lively curiosity about health care data and analytical tools is greatly desired. Position will work remote for now, applicants should live in the New York City area.\nJob Responsibilities:\nUnderstand the needs and opportunities across our internal and external stakeholders\nCreate, update, and distribute analytic reports on a recurring basis\nLearn new data related skills and processes\nLearn health care claim (UB04, HCFA 1500) formats and analysis\nShare and teach skills to others on the data team\nWork as part of a larger team and helping other teams in achieving their goals\nContinue to push the boundaries of what technology can do to empower our caregivers and clinicians to improve health outcomes for our patients\nRequirements:\nA College degree (required)\n5+ years experience working with health care data (e.g. claims, app output, CMS HCC, SDOH, clinical research) in queries or BI tools (required)\nExperience analyzing operational data and correlations it may have to clinical outcomes data\nMS Excel or Google Sheets advanced ability (pivot tables, formatting, graphics) (required)\nExperience with BI tools such as Tableau (highly desired)\nExperience with SQL, SAS, Python, SPSS or other analytic tools for data management and informatics\nAttention to detail along with a true love of data patterns, error detection and correction\nAbility to be a story-teller with data - to see and articulate clearly the story behind the numbers\nIntellectual curiosity around finding the why and digging deep to find patterns others might miss\nPassion about our mission to improve people s lives\nComfort in a dynamic and always evolving start-up environment\n\nMinimum Years of Experience: 5\n\nSub Specialties: Business Analyst- Associate\nRating: Not available\nCompany Name: SAT Healthcare\nLocation: New York, NY\nHeadquarters: Santa Clara, CA\nSize: 201 to 500 employees\nType of ownership: Company - Private\nIndustry: Health Care Services & Hospitals\nSector: Health Care\nCompetitors: -1', '_hybrid_score': 0.02976190476190476}
    {'id': 'job-04477', 'job_title': 'Senior Healthcare Data Analyst', 'salary_estimate': '$98K-$114K', 'job_description': "Senior Healthcare Data Analyst\n\nNew York, NY\n\nAre you a rock star healthcare data analyst who excels at solving challenging data problems and answering hard questions?\n\nDo you share our passion for enabling positive change within healthcare and helping patients with chronic conditions like diabetes, obesity, cardiovascular disease and mental health?\n\nIf so, you could be a perfect fit for our team of like-minded professionals who share a common mission and passion for helping others and a desire to build a great company.\n\nCecelia Health is a high-growth, venture-backed health tech services company based in New York City. Our customers are health plans, pharmaceutical & device companies, and self-insured employers who we work with to deliver personalized, technology-enabled coaching to transform the lives and health outcomes of their members, patients and employees. Cecelia Health is a high-energy, results-oriented work place that believes our success, as well as the success of our customers and patients, relies primarily on a fantastic team with the passion, drive and skills to change the face of chronic condition management.\n\nCecelia Health seeks to hire a Senior Healthcare Data Analyst to be based in the companys midtown location in New York City and report to our VP of Data & Strategy. The Senior Healthcare Data Analyst directly supports Cecelia Healths customer, operational and thought leadership needs by developing and designing standardized and customized analytics that support value based financial contracts, population health management, quality outcomes, performance targets, and identification of potential areas for improvement.\n\nWho You Are:\n\nYou are naturally curious, a strategic thinker and an effective communicator. We expect you to be well-versed in producing and interpreting custom and routine data analytics using membership, claims, provider, prescription drug and quality data from a variety of sources. You have high-accountability and can effectively work with various company stakeholders and make an impact across the company. If you are passionate about telling stories with data and welcome the challenge of answering vexing questions, we would love to meet you.\n\nResponsibilities:\nAnalyze, understand, interpret and explain complex membership, clinical, quality outcome and performance data in support of customer, sales marketing and market needs\nComplete assigned customer analytic projects by evaluating data requirements, designing analytics studies and delivering reports and results to customers and management\nEffectively identify, understand and communicate customer analytics and reporting business requirements\nEffectively communicate analytic findings with customers and management\nSupport customer Analytic Platform training and day to day support needs\nPerform routine data quality review and continuous quality improvement initiatives\nProactively mine customer-specific utilization, financial and clinical quality performance data trends to identify opportunities for enhanced analytics, clinical quality improvements and/or cost savings\nProvide recommendations for cost and clinical performance improvement to internal management and customers\nParticipate in continuous product improvement initiatives by identifying opportunities to improve customer analytics, dashboards and data platforms\nRequirements:\nBachelor's degree required in health care related, informatics, statistics or mathematics related disciplines\nMinimum of 5 years of direct experience in health care analytics field\nIn-depth knowledge of health insurance based provider billing, medical and pharmacy coding schemes, claims payment process, and claims, clinical, quality and eligibility data characteristics\nWorking knowledge of relational database structures, data quality issues and health care and pharmacy payer data elements\nDeep experience using Tableau and open source reporting packages\nExcellent verbal, written and presentation communication skills\nStrong analytic and problem solving skills\nSelf-motivated, intellectually curious, open learner\nMaster degree in healthcare or informatics (preferred)\nWere a young and rapidly growing Healthcare Technology company - if you have ever wanted to jump on a rocket ship as its taking off, now is your chance.\n\nPowered by JazzHR", 'rating': nan, 'company_name': 'Cecelia Health', 'location': 'New York, NY', 'headquarters': 'New York, NY', 'size': '51 to 200 employees', 'type_of_ownership': 'Company - Private', 'industry': 'Health Care Services & Hospitals', 'sector': 'Health Care', 'competitors': '-1', 'page_content': "Job Title: Senior Healthcare Data Analyst\nSalary Estimate: $98K-$114K\nJob Description: Senior Healthcare Data Analyst\n\nNew York, NY\n\nAre you a rock star healthcare data analyst who excels at solving challenging data problems and answering hard questions?\n\nDo you share our passion for enabling positive change within healthcare and helping patients with chronic conditions like diabetes, obesity, cardiovascular disease and mental health?\n\nIf so, you could be a perfect fit for our team of like-minded professionals who share a common mission and passion for helping others and a desire to build a great company.\n\nCecelia Health is a high-growth, venture-backed health tech services company based in New York City. Our customers are health plans, pharmaceutical & device companies, and self-insured employers who we work with to deliver personalized, technology-enabled coaching to transform the lives and health outcomes of their members, patients and employees. Cecelia Health is a high-energy, results-oriented work place that believes our success, as well as the success of our customers and patients, relies primarily on a fantastic team with the passion, drive and skills to change the face of chronic condition management.\n\nCecelia Health seeks to hire a Senior Healthcare Data Analyst to be based in the companys midtown location in New York City and report to our VP of Data & Strategy. The Senior Healthcare Data Analyst directly supports Cecelia Healths customer, operational and thought leadership needs by developing and designing standardized and customized analytics that support value based financial contracts, population health management, quality outcomes, performance targets, and identification of potential areas for improvement.\n\nWho You Are:\n\nYou are naturally curious, a strategic thinker and an effective communicator. We expect you to be well-versed in producing and interpreting custom and routine data analytics using membership, claims, provider, prescription drug and quality data from a variety of sources. You have high-accountability and can effectively work with various company stakeholders and make an impact across the company. If you are passionate about telling stories with data and welcome the challenge of answering vexing questions, we would love to meet you.\n\nResponsibilities:\nAnalyze, understand, interpret and explain complex membership, clinical, quality outcome and performance data in support of customer, sales marketing and market needs\nComplete assigned customer analytic projects by evaluating data requirements, designing analytics studies and delivering reports and results to customers and management\nEffectively identify, understand and communicate customer analytics and reporting business requirements\nEffectively communicate analytic findings with customers and management\nSupport customer Analytic Platform training and day to day support needs\nPerform routine data quality review and continuous quality improvement initiatives\nProactively mine customer-specific utilization, financial and clinical quality performance data trends to identify opportunities for enhanced analytics, clinical quality improvements and/or cost savings\nProvide recommendations for cost and clinical performance improvement to internal management and customers\nParticipate in continuous product improvement initiatives by identifying opportunities to improve customer analytics, dashboards and data platforms\nRequirements:\nBachelor's degree required in health care related, informatics, statistics or mathematics related disciplines\nMinimum of 5 years of direct experience in health care analytics field\nIn-depth knowledge of health insurance based provider billing, medical and pharmacy coding schemes, claims payment process, and claims, clinical, quality and eligibility data characteristics\nWorking knowledge of relational database structures, data quality issues and health care and pharmacy payer data elements\nDeep experience using Tableau and open source reporting packages\nExcellent verbal, written and presentation communication skills\nStrong analytic and problem solving skills\nSelf-motivated, intellectually curious, open learner\nMaster degree in healthcare or informatics (preferred)\nWere a young and rapidly growing Healthcare Technology company - if you have ever wanted to jump on a rocket ship as its taking off, now is your chance.\n\nPowered by JazzHR\nRating: Not available\nCompany Name: Cecelia Health\nLocation: New York, NY\nHeadquarters: New York, NY\nSize: 51 to 200 employees\nType of ownership: Company - Private\nIndustry: Health Care Services & Hospitals\nSector: Health Care\nCompetitors: -1", '_hybrid_score': 0.027692895339954164}



```python
hybrid_metrics = evaluate(
    gt_test,
    lambda q: hybrid_search(q["question"]),
)
print(hybrid_metrics)
```


      0%|          | 0/1146 [00:00<?, ?it/s]


    {'hit_rate': 0.6387434554973822, 'mrr': 0.4699970913321694}



```python
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
```


```python
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
```


      0%|          | 0/200 [00:00<?, ?it/s]



```python
def hybrid_rag(query, model="gpt-5.6-luna"):
    search_results = hybrid_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer
```


```python
df_eval = pd.DataFrame(evaluations, columns=["record", "answer", "evaluation"])

df_eval["id"] = df_eval.record.apply(lambda d: d["id"])
df_eval["question"] = df_eval.record.apply(lambda d: d["question"])
df_eval["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
df_eval["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])

df_eval.relevance.value_counts(normalize=True)
```




    relevance
    RELEVANT           0.835
    PARTLY_RELEVANT    0.110
    NON_RELEVANT       0.055
    Name: proportion, dtype: float64




```python
df_eval.to_csv("rag1-eval-gpt-5.6-luna.csv", index=False)
```


```python
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
```


      0%|          | 0/200 [00:00<?, ?it/s]



```python
df_eval1 = pd.DataFrame(evaluations_hybrid, columns=["record", "answer", "evaluation"])

df_eval1["id"] = df_eval.record.apply(lambda d: d["id"])
df_eval1["question"] = df_eval.record.apply(lambda d: d["question"])
df_eval1["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
df_eval1["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])

df_eval1.relevance.value_counts(normalize=True)
```




    relevance
    RELEVANT           0.835
    PARTLY_RELEVANT    0.110
    NON_RELEVANT       0.055
    Name: proportion, dtype: float64




```python
df_eval1.to_csv("hybrid-rag-eval-gpt-5.6-luna.csv", index=False)
```


```python

```


```python
jupyter nbconvert --to=script Untitled.ipynb
```


      Cell In[54], line 1
        jupyter nbconvert --to=script Untitled.ipynb
                ^
    SyntaxError: invalid syntax




```python

```


```python

```


```python

```


```python

```


```python
#uv add langchain langchain-elasticsearch langchain-huggingface sentence-transformers
```

    /home/justus/llm_projects/data_roles_assistant/.venv/bin/python: No module named uv
    Note: you may need to restart the kernel to use updated packages.



```python
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

es_url = "http://localhost:9200"
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/11.6k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]



    model.safetensors: reconstructing file:   0%|          |  0.00B / 90.9MB            



    model.safetensors: downloading bytes:           |  0.00B            



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]



    tokenizer_config.json:   0%|          | 0.00/383 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]



```python

```
