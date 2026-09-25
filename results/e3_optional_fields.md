# E3: does an optional field get omitted?

## Setup

Command: `python -m review_llm_eval.experiments run E3`

- `e3_free_text_optional_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e3_free_text_required_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
- `e3_free_text_optional_8b`: 2026-09-24, n=100, qwen3:8b digest `500a1f067a9f`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e3_free_text_required_8b`: 2026-09-24, n=100, qwen3:8b digest `500a1f067a9f`

The incident field in the shape of the first production contract: free text, `["string", "null"]`. `free_text_optional` leaves it out of `required`; `free_text_required` keeps it in. 100 reviews (20 per star).

| run | valid | key present | non-null when present | non-null overall |
| --- | --- | --- | --- | --- |
| e3_free_text_optional_4b | 100 | 0.000 | n/a | 0.000 |
| e3_free_text_required_4b | 100 | 1.000 | 0.450 | 0.450 |
| e3_free_text_optional_8b | 100 | 0.000 | n/a | 0.000 |
| e3_free_text_required_8b | 100 | 1.000 | 0.410 | 0.410 |

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e3_free_text_optional_4b | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 316.3 | 492.1 | 1017.9 | 0.48 | 10.1 | 31.8 | 21.6 |
| e3_free_text_required_4b | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 333.6 | 509.3 | 1017.9 | 0.52 | 11.2 | 31.9 | 20.1 |
| e3_free_text_optional_8b | 100 | 100 | 99 | 1 | 1 | 0 | 0 | 1 | 352.0 | 507.2 | 1023.9 | 0.55 | 15.1 | 23.6 | 14.4 |
| e3_free_text_required_8b | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 363.4 | 546.0 | 1023.9 | 0.72 | 15.3 | 24.1 | 14.5 |

