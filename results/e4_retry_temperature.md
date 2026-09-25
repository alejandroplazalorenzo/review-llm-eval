# E4: which temperature for the retry?

## Setup

Command: `python -m review_llm_eval.experiments run E4`

- `e1_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.2, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e10_repeat_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
- `e10_serial_a_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
- `e10_serial_b_4b`: 2026-09-23, n=100, qwen3:4b digest `359d7dd4bcda`
- `e5_numeric_4b`: 2026-09-23, n=200, qwen3:4b digest `359d7dd4bcda`
- `e1_8b`: 2026-09-24, n=200, qwen3:8b digest `500a1f067a9f`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7
- `e4_induced`: 2026-09-24, n=27, qwen3:4b digest `359d7dd4bcda`
- `e4_retry`: 2026-09-24, n=1, qwen3:8b digest `500a1f067a9f`

## Natural failures inside the runs (first attempt T=0, retry T=0.4, as in production)

Every run of every experiment; listed when a first attempt failed (plus the E4
sources). 'Other' failures are not cut at `num_predict`: in E6 they are the answers
that came back in the `thinking` field.

| run | reviews | 1st attempt invalid | cut at num_predict (runaway) | runaways rescued by the retry | other failures |
| --- | --- | --- | --- | --- | --- |
| e1_4b | 200 | 0 | 0 | 0 | 0 |
| e1_8b | 200 | 1 | 1 | 0 | 0 |
| e3_free_text_optional_8b | 100 | 1 | 1 | 1 | 0 |
| e5_numeric_4b | 200 | 0 | 0 | 0 | 0 |
| e6_think_omitted_4b | 24 | 24 | 0 | 0 | 24 |
| e6_think_true_4b | 24 | 24 | 0 | 0 | 24 |
| e6_think_false_8b | 24 | 1 | 1 | 0 | 0 |
| e6_think_omitted_8b | 24 | 3 | 3 | 0 | 0 |
| e6_think_true_8b | 24 | 3 | 3 | 1 | 0 |
| e8_8b_t1 | 48 | 1 | 1 | 1 | 0 |
| e8_8b_t2 | 48 | 1 | 1 | 0 | 0 |
| e8_8b_t4 | 48 | 1 | 1 | 0 | 0 |
| e8_8b_t8 | 48 | 1 | 1 | 1 | 0 |
| e10_repeat_4b | 200 | 0 | 0 | 0 | 0 |
| e10_serial_a_4b | 100 | 0 | 0 | 0 | 0 |
| e10_serial_b_4b | 100 | 0 | 0 | 0 | 0 |

| model | first attempts (all runs) | runaways | rescued by the T=0.4 retry |
| --- | --- | --- | --- |
| 4b | 2550 | 0 | 0 |
| 8b | 664 | 13 | 4 |

Reviews that ran away: review 21: 8 of 10 qwen3:8b runs, review 5649: 2 of 10 qwen3:8b runs, review 7064: 2 of 10 qwen3:8b runs, review 16670: 1 of 3 qwen3:8b runs.

## Retries of the failures seen in those runs

| retry temperature | calls | valid calls | share of calls valid | failures rescued by any call | calls identical to the failed answer |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0 | 0.000 | 0/1 | 1 |
| 0.4 | 5 | 0 | 0.000 | 0/1 | 0 |

Draws per failure: 2 at T=0, 5 at T=0.4, same prompt.

## Induced failures

`num_predict` lowered to 367 (75th percentile of the output tokens of qwen3:4b in E1) on the 27 reviews whose E1 answer was up to 20% longer. First attempt (T=0) invalid: 24 of 27 (cut at num_predict: 24).

These are complete answers cut short, not a loop like the natural runaways of
qwen3:8b (an opinion list that repeats the same items until `num_predict`): a
retry only succeeds if the model happens to write a shorter answer.

| retry temperature | calls | valid calls | share of calls valid | failures rescued by any call | calls identical to the failed answer |
| --- | --- | --- | --- | --- | --- |
| 0 | 48 | 1 | 0.021 | 1/24 | 24 |
| 0.4 | 120 | 18 | 0.150 | 7/24 | 6 |

Draws per failure: 2 at T=0, 5 at T=0.4, same prompt.

