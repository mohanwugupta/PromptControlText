# Credential verification

Verified on October 7, 2026 (America/Los_Angeles), using the local, Git-ignored
`.env` and Runpod CLI 2.14.0. This record contains no credential values, account
identifiers, balances, or resource details.

| Service / check | Result | What was established |
| --- | --- | --- |
| Runpod `user` | Passed | The key authenticates and can read the account. |
| Runpod `pod list --all` | Passed | The official CLI can list pods. |
| Runpod `GET /v1/pods` | HTTP 200 | REST pod listing succeeds. |
| OpenAI `GET /v1/models` | HTTP 200 | Authentication and model-list access succeed. |
| Anthropic `GET /v1/models` | HTTP 200 | Authentication and model-list access succeed. |
| Gemini `GET /v1beta/models` | HTTP 200 | The key can access the Gemini model catalogue. |
| Hugging Face `GET /api/whoami-v2` | HTTP 200 | The token authenticates. |
| Hugging Face Llama-3.1-8B-Instruct `auth-check` | HTTP 200 | The token can access the selected gated judge repository. |

All five credential variables were present. `.env` remained ignored by Git with
owner-only file permissions (`600`). Keys were loaded into process memory and
sent only to their corresponding services for these checks; their values were
not printed or included in this record.

The first bare Runpod REST request returned HTTP 403. The official CLI pod-list
request and a REST retry with an explicit User-Agent both succeeded. No key
permissions were changed; these observations do not establish the cause of the
initial 403.

## Scope and remaining checks

These were read-only access checks. No model inference, pod creation, model
weight download, or billing change was performed. Model listings alone do not
verify inference quota, credit availability, Batch access, or access to a
particular frontier model. Read-only Runpod success does not establish pod-write
permission. Those checks belong to the selected-model pilot and worker setup.

The planned study still uses one generator from each of OpenAI, Anthropic, and
Google, with `meta-llama/Llama-3.1-8B-Instruct` as the existing judge. No model
selection or experiment protocol was changed during credential verification.
