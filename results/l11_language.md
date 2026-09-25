# Language: model vs detector

## Setup

Command: `python -m review_llm_eval.experiments run L11`

- `l11_language_4b`: 2026-09-24, n=38, qwen3:4b digest `359d7dd4bcda`
  Ollama 0.34.3, GPU NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, OLLAMA_NUM_PARALLEL=4, OLLAMA_FLASH_ATTENTION=true, OLLAMA_KV_CACHE_TYPE=q8_0 (from server.log), Python 3.13.7

All 38 reviews of the dataset that the detector does not classify as Spanish (many are bilingual). Model = the `idi` field of qwen3:4b.

| detector | model | reviews |
| --- | --- | --- |
| de | es | 1 |
| en | es | 36 |
| pt | pt | 1 |

Model agrees with the detector on 1 of 38. Of the 11 reviews without a single Spanish function word, the model answered `es` for 11.

