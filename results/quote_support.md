# Quote support distribution (threshold derivation)

## Setup

Command: `python -m review_llm_eval.experiments analyze all`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e1_8b`: 2026-09-24, n=200, qwen3:8b digest `500a1f067a9f`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

Support = share of a quote's content words (4+ letters) found in the review;
1.0 = literal after folding case, accents and spacing. Fillers excluded.

| support | qwen3:4b | qwen3:8b |
| --- | --- | --- |
| [0.0, 0.1) | 4 | 1 |
| [0.1, 0.2) | 0 | 0 |
| [0.2, 0.3) | 1 | 0 |
| [0.3, 0.4) | 1 | 0 |
| [0.4, 0.5) | 0 | 0 |
| [0.5, 0.6) | 0 | 0 |
| [0.6, 0.7) | 2 | 0 |
| [0.7, 0.8) | 1 | 1 |
| [0.8, 0.9) | 1 | 3 |
| [0.9, 1.0) | 2 | 10 |
| 1.0 (literal) | 673 | 834 |

