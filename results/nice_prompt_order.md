# Instructions first (prefix caching)

## Setup

Command: `python -m review_llm_eval.experiments run NICE`

- `e10_repeat_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
- `nice_reorder_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`

## Reliability and speed

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.2 | 474.6 | 1108.3 | 0.50 | 11.1 | 29.6 | 19.1 |
| e1_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.5 | 480.2 | 1108.3 | 0.55 | 9.8 | 32.9 | 22.3 |
| nice_reorder_4b | 200 | 196 | 196 | 0 | 0 | 0 | 4 | 0 | 337.4 | 502.2 | 1109.1 | 0.22 | 10.0 | 35.2 | 21.1 |

## Identical answers per field (share of reviews valid in both runs)

| comparison | pairs | json | opinions | quotes | staff | incident | says_no_return | legal_action | theft | illness | bad_faith | returning_guest | with_children | recommends | nights | language | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b vs nice_reorder_4b | 196 | 0.00 | 0.16 | 0.07 | 0.83 | 0.80 | 0.79 | 0.99 | 0.99 | 1.00 | 0.99 | 0.93 | 0.96 | 0.91 | 0.92 | 1.00 | 0.01 |
| e1_4b vs e10_repeat_4b | 200 | 0.36 | 0.80 | 0.74 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 0.42 |

`prompt tok` counts the whole prompt even when the server reuses a cached prefix;
the saving shows in `prompt eval s` (mean prompt-processing time of the first
attempt). Second row: repeatability baseline.

## How often each field fires

Counts over valid outputs, after post-processing. `_no_phrase` = raised on a review
that contains no phrase for that alert (`cues.py`, deliberately broad).

| run | valid | says_no_return | says_no_return_no_phrase | bad_faith | bad_faith_no_phrase | theft | illness | legal_action | recommends | returning_guest | with_children | incident | opinions | staff_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 84 | 59 | 10 | 1 | 3 | 2 | 0 | 71 | 15 | 25 | 52 | 617 | 170 |
| e1_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 72 | 15 | 25 | 52 | 614 | 166 |
| nice_reorder_4b | 196 | 41 | 18 | 9 | 1 | 4 | 2 | 1 | 57 | 25 | 26 | 27 | 690 | 182 |

