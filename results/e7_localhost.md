# E7: localhost or 127.0.0.1?

## Setup

Command: `python -m review_llm_eval.experiments run E7`

- `e7_ip_4b`: 2026-09-23, n=50, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e7_localhost_4b`: 2026-09-23, n=50, qwen3:4b digest `359d7dd4bcda`

50 reviews, 1 thread, same everything except the host name in the URL.

| run | calls | client s/call | server s/call | overhead s/call | reviews/min |
| --- | --- | --- | --- | --- | --- |
| e7_ip_4b | 50 | 4.782 | 4.748 | 0.033 | 12.546 |
| e7_localhost_4b | 50 | 6.060 | 3.985 | 2.075 | 9.899 |

