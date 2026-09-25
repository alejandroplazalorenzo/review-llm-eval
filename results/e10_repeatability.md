# E10: is temperature 0 repeatable?

## Setup

Command: `python -m review_llm_eval.experiments run E10`

- `e10_repeat_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e10_serial_a_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
- `e10_serial_b_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`

## Reliability and speed

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.2 | 474.6 | 1108.3 | 0.50 | 11.1 | 29.6 | 19.1 |
| e10_serial_a_4b | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 327.3 | 521.0 | 1108.9 | 0.41 | 5.0 | 70.6 | 11.4 |
| e10_serial_b_4b | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 325.7 | 521.0 | 1108.9 | 0.41 | 5.2 | 69.0 | 11.2 |
| e1_4b | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 315.5 | 480.2 | 1108.3 | 0.55 | 9.8 | 32.9 | 22.3 |

## Identical answers per field (share of reviews valid in both runs)

| comparison | pairs | json | opinions | quotes | staff | incident | says_no_return | legal_action | theft | illness | bad_faith | returning_guest | with_children | recommends | nights | language | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b vs e10_repeat_4b | 200 | 0.36 | 0.80 | 0.74 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 0.42 |
| e10_serial_a_4b vs e10_serial_b_4b | 100 | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 |

Row 1: 200 reviews twice with 4 threads.
Row 2: 100 reviews twice, one request at a time.

## How often each field fires

Counts over valid outputs, after post-processing. `_no_phrase` = raised on a review
that contains no phrase for that alert (`cues.py`, deliberately broad).

| run | valid | says_no_return | says_no_return_no_phrase | bad_faith | bad_faith_no_phrase | theft | illness | legal_action | recommends | returning_guest | with_children | incident | opinions | staff_names |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e10_repeat_4b | 200 | 84 | 59 | 10 | 1 | 3 | 2 | 0 | 71 | 15 | 25 | 52 | 617 | 170 |
| e10_serial_a_4b | 100 | 42 | 29 | 4 | 1 | 1 | 0 | 0 | 35 | 9 | 13 | 22 | 317 | 106 |
| e10_serial_b_4b | 100 | 42 | 29 | 4 | 1 | 1 | 0 | 0 | 36 | 9 | 13 | 22 | 316 | 105 |
| e1_4b | 200 | 85 | 60 | 10 | 1 | 3 | 2 | 0 | 72 | 15 | 25 | 52 | 614 | 166 |

