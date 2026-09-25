# maxLength on quotes and summary

## Setup

Command: `python -m review_llm_eval.experiments run NICE`

- `e10_repeat_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
- `nice_maxlength_4b`: 2026-09-24, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

## Reliability and speed

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.2 | 474.6 | 1108.3 | 0.50 | 11.1 | 29.6 | 19.1 |
| e1_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.5 | 480.2 | 1108.3 | 0.55 | 9.8 | 32.9 | 22.3 |
| nice_maxlength_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 312.9 | 475.3 | 1108.3 | 0.47 | 10.4 | 31.6 | 21.6 |

## Identical answers per field (share of reviews valid in both runs)

| comparison | pairs | json | opinions | quotes | staff | incident | says_no_return | legal_action | theft | illness | bad_faith | returning_guest | with_children | recommends | nights | language | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b vs nice_maxlength_4b | 200 | 0.32 | 0.77 | 0.63 | 0.96 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.99 | 1.00 | 0.35 |
| e1_4b vs e10_repeat_4b | 200 | 0.36 | 0.80 | 0.74 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 0.42 |

maxLength (characters): quote 225, summary 351 = 95th
percentile of the lengths written in e1_4b. Second row: repeatability baseline.

## How often each field fires

Counts over valid outputs, after post-processing. `_no_phrase` = raised on a review
that contains no phrase for that alert (`cues.py`, deliberately broad).

| run | valid | says_no_return | says_no_return_no_phrase | bad_faith | bad_faith_no_phrase | theft | illness | legal_action | recommends | returning_guest | with_children | incident | opinions | staff_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 84 | 59 | 10 | 1 | 3 | 2 | 0 | 71 | 15 | 25 | 52 | 617 | 170 |
| e1_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 72 | 15 | 25 | 52 | 614 | 166 |
| nice_maxlength_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 71 | 14 | 25 | 50 | 623 | 172 |

