# E9: is keep_alive needed?

## Setup

Command: `python -m review_llm_eval.experiments run E9`

- `e9_keep_alive`: 2026-09-24, n=4, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

qwen3:4b, one review, 1 thread. Before each timed call the model is used once and then left idle for 330 s (the server default keep_alive is 5 min).

| rep | keep_alive | model loaded before the call | latency s | load s | at |
| --- | --- | --- | --- | --- | --- |
| 0 | omitted | none | 11.432 | 6.111 | 2026-09-24T14:55:13+00:00 |
| 0 | 30m | qwen3:4b | 5.240 | 0.020 | 2026-09-24T15:00:54+00:00 |
| 1 | omitted | qwen3:4b | 5.093 | 0.014 | 2026-09-24T15:06:34+00:00 |
| 1 | 30m | qwen3:4b | 5.049 | 0.037 | 2026-09-24T15:12:14+00:00 |

Rep(s) 1: the call without `keep_alive` came right after a request with `keep_alive: 30m`, and the model was still loaded after 330 s. A request that leaves `keep_alive` out did not shorten the timer an earlier request had set, so only the other rep(s) of `omitted` measure a reload after the default 5 minutes.

