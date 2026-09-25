# Incident field: three alternative wordings

## Setup

Command: `python -m review_llm_eval.experiments run NICE`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `nice_inc_no_default_4b`: 2026-09-23, n=120, qwen3:4b digest `359d7dd4bcda`
- `nice_inc_phrases_4b`: 2026-09-23, n=120, qwen3:4b digest `359d7dd4bcda`
- `nice_inc_list_4b`: 2026-09-23, n=120, qwen3:4b digest `359d7dd4bcda`

120 reviews (24 per star). 'A phrase' = the review contains an incident phrase (`cues.py`, deliberately broad).

| run | valid | incident on 1-2 star | incident on 5 star | incident when a phrase is present | incident without any phrase |
| --- | --- | --- | --- | --- | --- |
| e1_4b | 120 | 0.396 | 0.000 | 0.538 | 11 |
| nice_inc_no_default_4b | 120 | 0.479 | 0.000 | 0.641 | 15 |
| nice_inc_phrases_4b | 120 | 0.312 | 0.000 | 0.462 | 7 |
| nice_inc_list_4b | 120 | 0.854 | 0.000 | 0.795 | 40 |

