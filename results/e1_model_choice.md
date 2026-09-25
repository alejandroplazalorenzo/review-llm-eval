# E1: qwen3:4b or qwen3:8b?

## Setup

Command: `python -m review_llm_eval.experiments run E1`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e1_8b`: 2026-09-24, n=200, qwen3:8b digest `500a1f067a9f`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

- qwen3:4b: Qwen3-4B, 2507 'thinking' release (declares 262,144 context; its chat template always opens <think> and has no branch for think=false)
- qwen3:8b: Qwen3-8B, original release (declares 40,960 context; its template adds /no_think and an empty <think></think> when think=false)

Sample: 200 reviews, 40 per star rating; 4 client threads, OLLAMA_NUM_PARALLEL=4; production settings.

## Reliability and speed

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.5 | 480.2 | 1108.3 | 0.55 | 9.8 | 32.9 | 22.3 |
| e1_8b | 200 | 199 | 199 | 1 | 0 | 1 | 0 | 1 | 347.3 | 496.0 | 1114.3 | 1.12 | 16.4 | 22.7 | 13.7 |

Thread curve (1/2/4/8 threads, 48 reviews): see `e8_threads.md`.

## The ten objective checks (`checks.py`, fixed before the qwen3:8b run)

| check | what | qwen3:4b | qwen3:8b | better |
| --- | --- | --- | --- | --- |
| valid_first_attempt | share of reviews with a schema-valid first answer | 1.000 | 0.995 | 4b |
| runaway | share of reviews with an attempt cut at num_predict | 0.000 | 0.005 | 4b |
| duplicate_pairs | share of valid outputs repeating a (topic, polarity) pair | 0.190 | 0.357 | 4b |
| name_not_in_text | share of staff names not written in the review | 0.000 | 0.011 | 4b |
| role_as_name | staff entries that are a job title or generic word (count) | 5 | 0 | 8b |
| unsupported_quote | share of quotes below the support threshold | 0.009 | 0.001 | 8b |
| filler_quote | share of quotes that are placeholders | 0.000 | 0.002 | 4b |
| no_return_5star_no_cue | 5-star reviews flagged 'will not return' without any such phrase (count) | 0 | 0 | tie |
| legal_no_cue | reviews flagged 'legal action' without any such phrase (count) | 0 | 1 | 4b |
| never_varies | output fields with one single value over the sample (count) | 2 | 1 | 8b |

qwen3:4b better or equal in **7 of 10**; qwen3:8b better or equal in **4 of 10**.

Fields that never varied: 4b `legal_action,language`, 8b `language`. Staff names (job titles excluded): 166 / 175, of which not in the review 0 / 2. Quotes: 685 / 851, of which fillers 0 / 2 and unsupported 6 / 1.

## How often each field fires (not one of the ten checks)

Counts over valid outputs, after post-processing. `_no_phrase` = raised on a
review that contains no phrase for that alert (`cues.py`, deliberately broad).

| run | valid | says_no_return | says_no_return_no_phrase | bad_faith | bad_faith_no_phrase | theft | illness | legal_action | recommends | returning_guest | with_children | incident | opinions | staff_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 72 | 15 | 25 | 52 | 614 | 166 |
| e1_8b | 199 | 66 | 42 | 15 | 5 | 4 | 2 | 1 | 52 | 20 | 35 | 61 | 721 | 173 |

`says_no_return` per star rating: raised / valid outputs (of those, raised on a
review with no such phrase). The check above only looks at 5-star reviews.

| run | 1 star | 2 star | 3 star | 4 star | 5 star |
| --- | --- | --- | --- | --- | --- |
| e1_4b | 40/40 (34) | 37/40 (22) | 8/40 (4) | 0/40 (0) | 0/40 (0) |
| e1_8b | 35/40 (29) | 27/40 (12) | 4/40 (1) | 0/40 (0) | 0/39 (0) |

