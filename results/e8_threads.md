# E8: how many client threads?

## Setup

Command: `python -m review_llm_eval.experiments run E8`

- `e8_4b_t1`: 2026-09-23, n=48, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e8_4b_t2`: 2026-09-23, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_4b_t4`: 2026-09-23, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_4b_t8`: 2026-09-23, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_8b_t1`: 2026-09-24, n=48, qwen3:8b digest `500a1f067a9f`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e8_8b_t2`: 2026-09-24, n=48, qwen3:8b digest `500a1f067a9f`
- `e8_8b_t4`: 2026-09-24, n=48, qwen3:8b digest `500a1f067a9f`
- `e8_8b_t8`: 2026-09-24, n=48, qwen3:8b digest `500a1f067a9f`
- `e8_4b_t1_b`: 2026-09-24, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_4b_t2_b`: 2026-09-24, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_4b_t4_b`: 2026-09-24, n=48, qwen3:4b digest `359d7dd4bcda`
- `e8_4b_t8_b`: 2026-09-24, n=48, qwen3:4b digest `359d7dd4bcda`

48 reviews per run, server OLLAMA_NUM_PARALLEL=4, batches of 25 as in the pipeline.

| run | n | valid | valid 1st | retried | rescued | failed | net err | runaway | out tok mean | out tok p95 | prompt tok | prompt eval s | latency p50 s | decode tok/s | reviews/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e8_4b_t1 | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 313.8 | 471.4 | 1104.2 | 0.05 | 4.5 | 72.0 | 12.9 |
| e8_4b_t2 | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 313.8 | 471.4 | 1104.2 | 0.07 | 10.8 | 30.5 | 10.6 |
| e8_4b_t4 | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 314.1 | 463.6 | 1104.2 | 0.52 | 10.9 | 31.2 | 21.0 |
| e8_4b_t8 | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 316.6 | 463.6 | 1104.2 | 0.08 | 17.3 | 35.1 | 25.2 |
| e8_8b_t1 | 48 | 48 | 47 | 1 | 1 | 0 | 0 | 1 | 353.2 | 482.0 | 1110.2 | 0.02 | 7.8 | 46.4 | 7.1 |
| e8_8b_t2 | 48 | 47 | 47 | 1 | 0 | 1 | 0 | 1 | 351.2 | 482.0 | 1110.2 | 0.04 | 12.4 | 29.1 | 8.9 |
| e8_8b_t4 | 48 | 47 | 47 | 1 | 0 | 1 | 0 | 1 | 353.1 | 480.7 | 1110.2 | 0.81 | 15.8 | 22.5 | 13.1 |
| e8_8b_t8 | 48 | 48 | 47 | 1 | 1 | 0 | 0 | 1 | 354.8 | 480.7 | 1110.2 | 0.06 | 27.3 | 25.2 | 15.2 |
| e8_4b_t1_b | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 315.5 | 488.0 | 1104.2 | 0.05 | 4.8 | 75.0 | 12.0 |
| e8_4b_t2_b | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 317.0 | 482.4 | 1104.2 | 0.44 | 9.9 | 31.9 | 11.4 |
| e8_4b_t4_b | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 317.7 | 481.1 | 1104.2 | 0.08 | 10.4 | 32.8 | 21.7 |
| e8_4b_t8_b | 48 | 48 | 48 | 0 | 0 | 0 | 0 | 0 | 321.8 | 495.4 | 1104.2 | 0.08 | 18.1 | 35.2 | 24.0 |

