# E5: numeric word limit or a description?

## Setup

Command: `python -m review_llm_eval.experiments run E5`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e5_numeric_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`

`words`: summary length described in words (production). `numeric`: "un resumen de N palabras como máximo" with N = 30 for 1-3 stars and 15 for 4-5. 'Over the numeric limit' applies the same N to
both runs. Echo = the summary mentions words or a maximum.

| run | n | valid | 1st attempt cut at num_predict | echo in valid summaries | echo in 1st raw answer | summary words mean | summary words p95 | summary words max | over the numeric limit | out tok mean | out tok p95 | out tok max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e1_4b | 200 | 200 | 0 | 1 | 1 | 33.900 | 61.100 | 110 | 164 | 315.510 | 480.200 | 628 |
| e5_numeric_4b | 200 | 200 | 0 | 2 | 2 | 27.735 | 45.000 | 58 | 133 | 307.945 | 456.400 | 578 |

