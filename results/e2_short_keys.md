# E2: short or long keys?

## Setup

Command: `python -m review_llm_eval.experiments run E2`

- `e10_repeat_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
- `e2_long_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`

## Reliability and speed

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.2 | 474.6 | 1108.3 | 0.50 | 11.1 | 29.6 | 19.1 |
| e1_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.5 | 480.2 | 1108.3 | 0.55 | 9.8 | 32.9 | 22.3 |
| e2_long_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 330.9 | 483.6 | 1129.3 | 0.63 | 11.4 | 31.2 | 19.8 |

## Identical answers per field (share of reviews valid in both runs)

| comparison | pairs | json | opinions | quotes | staff | incident | says_no_return | legal_action | theft | illness | bad_faith | returning_guest | with_children | recommends | nights | language | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b vs e2_long_4b | 200 | n/a | 0.39 | 0.20 | 0.90 | 0.90 | 0.83 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 | 0.95 | 1.00 | 0.01 |
| e1_4b vs e10_repeat_4b | 200 | 0.36 | 0.80 | 0.74 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 0.42 |

The second row is the repeatability baseline (same prompt twice, E10): a variant
only 'changes the answers' where it falls below that line.

## How often each field fires

Counts over valid outputs, after post-processing. `_no_phrase` = raised on a review
that contains no phrase for that alert (`cues.py`, deliberately broad).

| run | valid | says_no_return | says_no_return_no_phrase | bad_faith | bad_faith_no_phrase | theft | illness | legal_action | recommends | returning_guest | with_children | incident | opinions | staff_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 84 | 59 | 10 | 1 | 3 | 2 | 0 | 71 | 15 | 25 | 52 | 617 | 170 |
| e1_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 72 | 15 | 25 | 52 | 614 | 166 |
| e2_long_4b | 200 | 51 | 27 | 11 | 2 | 3 | 1 | 0 | 74 | 16 | 27 | 62 | 616 | 179 |

