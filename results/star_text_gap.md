# Stars vs text

Pipeline rows with version `pysentimiento-robertuito+langdet-v1/v1+qwen3:4b/prompt-v1`: 200. A review is listed when layer 1
(pysentimiento) and the LLM opinions both contradict the star rating.

| case | reviews | review ids (row index in the public parquet) |
| --- | --- | --- |
| high stars negative text | 4 | 13670, 19618, 28814, 29600 |
| low stars positive text | 0 |  |
