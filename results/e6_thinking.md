# E6: thinking and `format`

## Setup

Command: `python -m review_llm_eval.experiments run E6`

- `e6_think_false_4b`: 2026-09-24, n=24, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e6_think_omitted_4b`: 2026-09-24, n=24, qwen3:4b digest `359d7dd4bcda`
- `e6_think_true_4b`: 2026-09-24, n=24, qwen3:4b digest `359d7dd4bcda`
- `e6_think_false_8b`: 2026-09-24, n=24, qwen3:8b digest `500a1f067a9f`
- `e6_think_omitted_8b`: 2026-09-24, n=24, qwen3:8b digest `500a1f067a9f`
- `e6_think_true_8b`: 2026-09-24, n=24, qwen3:8b digest `500a1f067a9f`

24 reviews, 1 thread, `format` = schema in all runs; only `think` changes.

| run | n | valid 1st | valid | runaway | out tok mean | latency 1st s | response chars mean | answers with thinking | thinking chars mean | thinking identical in size to the think=false answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e6_think_false_4b | 24 | 24 | 24 | 0 | 330.500 | 4.187 | 1006.792 | 0 | 0.000 | 0 |
| e6_think_omitted_4b | 24 | 0 | 0 | 0 | 329.083 | 4.504 | 0.000 | 24 | 1002.833 | 13 |
| e6_think_true_4b | 24 | 0 | 0 | 0 | 329.083 | 4.600 | 0.000 | 24 | 1002.833 | 13 |
| e6_think_false_8b | 24 | 23 | 23 | 1 | 363.708 | 9.460 | 1102.625 | 0 | 0.000 | 0 |
| e6_think_omitted_8b | 24 | 21 | 21 | 3 | 377.625 | 9.868 | 1241.000 | 0 | 0.000 | 0 |
| e6_think_true_8b | 24 | 21 | 22 | 3 | 366.500 | 9.534 | 1204.208 | 0 | 0.000 | 0 |

Last column: first attempts whose `thinking` field has exactly the length (in
characters) and the token count of the same model's think=false answer to the same
review, i.e. the JSON answer delivered in the `thinking` field.

