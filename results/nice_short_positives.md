# Skipping short positive reviews

## Setup

Command: `python -m review_llm_eval.experiments analyze all`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

'Short' = 4-5 star reviews under 241 characters (the 25th percentile of 4-5 star lengths); they are 21.8% of the 33038 reviews of the dataset. The dataset has no review under 60 characters, so this is not the same cut as production.

| group | valid | with a staff name | recommends | with an alert | opinions per review | latency s (4 threads) |
| --- | --- | --- | --- | --- | --- | --- |
| short (< p25 length) | 22 | 0.500 | 0.955 | 0 | 2.500 | 7.632 |
| the other 4-5 star | 58 | 0.586 | 0.862 | 0 | 3.414 | 10.209 |

