# review-llm-eval

A rebuild, on public data, of the review-enrichment pipeline I run at work on
automotive dealer reviews, plus the experiments that re-measure each of its decisions.
Every review goes through two layers:

1. **Layer 1, no LLM:** sentiment and emotion with
   [pysentimiento](https://github.com/pysentimiento/pysentimiento) (RoBERTuito), the
   language with a deterministic detector, the date precision from metadata.
2. **Layer 2, a local LLM** (`qwen3:4b` in Ollama) under a JSON Schema: opinions per
   topic with a literal quote, staff names, an incident code, explicit-only alert flags,
   a few guest qualifiers and a short summary in Spanish.

Deterministic post-processing repairs what code can decide, quality gates check the
stored output after every run, and experiments E1-E10 re-measure the production
decisions on 33,038 Spanish hotel reviews
([`beltrewilton/punta-cana-spanish-reviews`](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews), MIT).

The README keeps two things apart: the **decisions** I took in production (stated
without internal figures) and what this repository **re-measured on public data**
(every number below comes from a file in [`results/`](results/), which gives the
command, n, date, Ollama version and model digest of each run). Where the public data
disagrees with what I saw in production, the table says so.

## At a glance: qwen3:4b vs qwen3:8b

Same contract, same prompt, same 200 reviews (40 per star rating), production settings,
RTX 3070 Laptop GPU (8 GB). Sources: [`results/e1_model_choice.md`](results/e1_model_choice.md)
and [`results/e8_threads.md`](results/e8_threads.md).

| | qwen3:4b | qwen3:8b |
| --- | --- | --- |
| Schema-valid answer at the first attempt | 200 / 200 | 199 / 200 (1 cut at `num_predict`, not rescued) |
| Reviews per minute, 4 client threads | **22.3** | 13.7 |
| Reviews per minute, 1 thread | 12.9 / 12.0 (two passes) | 7.1 |
| Decode speed of one request alone (tok/s) | 72 / 75 | 46 |
| Output tokens per review (mean) | 316 | 347 |
| Valid outputs repeating a (topic, polarity) pair | 19.0 % | 35.7 % |
| Staff names not written in the review | 0 of 166 | 2 of 175 |
| Job titles returned as staff names | 5 | 0 |
| Quotes not supported by the review (before post-processing) | 6 of 685 | 1 of 849 |
| Objective checks where the model is better or equal (of 10) | **7** | 4 |
| (topic, polarity) F1 against a stronger model's blind labels, 100 reviews | 0.63 | **0.66** |

The two tags are **different generations**: the local `qwen3:4b` (digest
`359d7dd4bcda`) is the 2507 "thinking" release, the `qwen3:8b` (`500a1f067a9f`) is the
original Qwen3 release. The comparison is between those two artefacts as Ollama ships
them, not between two sizes of one model.

## Architecture

```
Hugging Face parquet (34,561 rows)
   │ fetch.py
   ▼
ingest.py: normalise, drop exact repeats (33,038 left), month from `wrote`
   │                                   ──► SQLite `review` (source data only)
   ▼
enrich.py ─ layer 1 (layer1.py): pysentimiento sentiment + emotion, langdetect.py
   │        ──► `review_enrichment` (version = layer 1)
   ├─ layer 2 (extract.py, 4 threads, batches of 25, commit per batch)
   │    prompt.py   one prompt: review block (rating, title, text) then instructions
   │    client.py   POST 127.0.0.1:11434/api/generate  format=<schema> think=false
   │                keep_alive=30m  temperature 0, retry at 0.4  num_ctx 4096
   │                num_predict 1024
   │    contract.py short keys, every key required, re-validated with jsonschema
   │    postprocess.py  expand codes, de-duplicate, drop filler / invented quotes,
   │                    check staff names against the text, veto roles and hotels
   │    ──► `review_enrichment` (version = layer1+model/prompt, raw output kept)
   └─ gates.py  quality gates on the stored rows ──► results/gates.md

experiments.py (runner.py, checks.py, analysis.py) ──► data/experiments/ ──► results/*.md
label.py / evaluate.py / alerts_review.py: human labels only (gold/)
```

Plain Python (`requests`, `jsonschema`, `pyarrow`, `sqlite3`); `pysentimiento` is an
optional extra for layer 1. Metrics are implemented by hand in `metrics.py`.

## Decisions from my production system (no internal figures)

What the production pipeline does and why, as I decided it on the real data. The
numbers behind these decisions stay private; the last column says where this repository
measures the same question again.

| Decision | Why | Re-measured here |
| --- | --- | --- |
| Overall sentiment from a dedicated classifier (pysentimiento), not from the LLM; irony detection dropped | Cheaper, runs on CPU, does not move when the prompt changes; the irony classifier gave false positives on ordinary reviews | layer 1 of the pipeline (not an experiment) |
| Language from a deterministic word-list detector; the model's answer kept only in the raw output | The model under-counted reviews written in other languages | L11 |
| Date precision and business line come from metadata; metadata overrides the model's area where the location is not a dealership | The source knows them; the model can only guess | month from `wrote`; no area field here |
| `qwen3:4b` over `qwen3:8b` | Measured on a sample of real reviews: faster, and better or equal on objective checks. Only some of those checks were written down at the time, so E1 writes ten down in code and applies them to both models | E1, E8 |
| `/api/generate`, one prompt, review first; `format` = JSON Schema, every answer re-validated with `jsonschema`; `think: false` | Constrained decoding, never trusted blindly; a thinking model spends tokens before the JSON | E6 |
| Short keys and short enum codes, expanded in Python | Output tokens are the cost of every call | E2 |
| Every key required, nullable only by type | An optional key under constrained decoding stopped being emitted and a whole column stayed NULL | E3 |
| Two attempts, same prompt, temperature 0 then 0.4; a row still invalid stays pending with a warning | At temperature 0 the retry repeated the failure (a string that runs until `num_predict`) | E4 |
| Summary length described in words and tied to the rating | A numeric word limit was echoed inside the summary and caused runaways | E5 |
| `127.0.0.1`, not `localhost` | On Windows `localhost` tried IPv6 first and every call paid for it | E7 |
| 4 client threads with `OLLAMA_NUM_PARALLEL=4` (flash attention, `q8_0` KV cache) | Generation is memory-bandwidth bound; parallel requests amortise it up to a point | E8 |
| `keep_alive: "30m"` | Without it the model was unloaded after 5 idle minutes and the next call paid for the reload | E9 |
| Deterministic post-processing: de-duplicate (topic, sentiment), drop filler quotes and quotes not found in the text, staff names must be written in the review (never taken from the business reply), job titles and brand names vetoed | Asking the model costs tokens and guarantees nothing; code does it for sure. Thresholds sit in an empty gap of the observed distribution | E1 checks, quote support |
| Alerts are booleans that need an explicit statement; precision over recall; calibrated by reading small batches by hand | An alert feeds a person's inbox: a false one costs more than a missed one | E1, gates, `alerts_review.py` |
| Incident enum defaults to "none"; high precision, low recall on purpose | Rewording did not fix the recall, and the variant that did made it fire on happy reviews | incident wording |
| Short positive reviews are not skipped | Their opinions are filler and they never raise an alert, but some of them name an employee | short positives |
| Model version and prompt version on every row, content hash, raw output stored, `--reprocess` | A mapping change can be re-applied without the GPU; a new prompt redoes only old rows | `enrich --remap`, `--reprocess` |
| Commit per batch, "pending" is a query, abort after 5 network errors in a row, a failed health check falls back to layer 1 | Backfills run for hours on a shared desktop | pipeline tests |
| Quality gates on the stored output: a field that never varies is broken; no duplicates; literal quotes; names in the text; no role as a name; few alerts on 4-5 stars | Each one watches a failure that passed the schema in some version | `gates.py`, run after every `enrich` |
| Golden checks are hand-verified aggregates, not LLM labels; there is no human gold set for opinions yet | A model cannot measure how often a model is wrong | golden gate; `label.py` (pending, see below) |
| The rating goes into the prompt | It is context the customer gave, and it sets the summary length | prompt |
| Business-reply judgement (four fields, null when there is no reply) | The block had to start with the "there is a reply" case or the model anchored on null | not reproducible: the dataset has no replies |

## Re-measured here on public data

**Setup.** RTX 3070 Laptop GPU (8 GB), Windows 11, Ollama 0.34.2 (runs of 23 Sep 2026)
and 0.34.3 (runs of 24 Sep 2026), server with `OLLAMA_NUM_PARALLEL=4`,
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`. Model digests `359d7dd4bcda`
(qwen3:4b) and `500a1f067a9f` (qwen3:8b). Main sample: 200 reviews, 40 per star rating
(seed 42), so rates are not rates of the natural distribution (88 % of the dataset is
4-5 stars). Each experiment writes one file in `results/`; `results/runs.csv` lists every
run with its date, n, digest and Ollama version.

**Noise floor.** The same run repeated (E10, 4 threads) gave 22.3 and 19.1 reviews/min:
this laptop GPU ends the runs at 78-89 °C (median 88 °C, recorded in `results/runs.csv`)
and throttles, so throughput differences under about 15 % between runs are not evidence
of anything. The second pass of the thread curve,
on the newer Ollama version (21.7 vs 21.0 reviews/min at 4 threads), shows that the
version change between the two days did not move the speed.

| # | Lesson from production | Here | Evidence |
| --- | --- | --- | --- |
| E1 | A smaller model can be faster and better | **Faster: reproduced. Better: mixed** | qwen3:4b: 22.3 vs 13.7 reviews/min (1.6×), 200/200 valid vs 199/200, better or equal on 7 of 10 objective checks (the 8b: 4 of 10; the checks were fixed after the 4b run and before the 8b run). The 4b halves the duplicated opinions (19.0 % vs 35.7 %) but returns more job titles as names (5 vs 0) and more unsupported quotes (6 vs 1). Against a stronger model's blind labels the 8b finds a few more opinions and names (pair F1 0.66 vs 0.63): the 4b is a speed choice. [e1](results/e1_model_choice.md), [judge](results/judge_agreement.md) |
| E2 | Output tokens are the cost: short keys | **Direction reproduced, size not** | Long keys: 331 vs 316 output tokens (+5 %), 19.8 vs 22.3 reviews/min, inside the noise floor. Here the long names are English identifiers that take few tokens, so there is little to save. The long keys also changed the answers: identical opinion sets in 39 % of reviews against 80 % for a plain repeat, and the no-return alert fired 51 times instead of 85. [e2](results/e2_short_keys.md) |
| E3 | Optional keys disappear under constrained decoding | **Reproduced, both models** | Free-text incident left out of `required`: the key appeared in **0 of 100** answers with qwen3:4b and 0 of 100 with qwen3:8b. Kept in `required`, it appeared in all of them and was non-null in 45 % (4b) and 41 % (8b). [e3](results/e3_optional_fields.md) |
| E4 | A retry at T=0 repeats the failure; with temperature it is rescued | **Partly** | qwen3:4b never ran away (0 of 2,550 first attempts cut at `num_predict`, all runs). qwen3:8b did 13 times in 664, 8 of them on the same review, which ran away in 8 of the 10 qwen3:8b runs that included it (all but the two E3 variants); the production retry (one call at 0.4) rescued 4 of the 13. Failures induced on qwen3:4b by lowering `num_predict` to 367 (24 of 27 reviews cut): re-asked at T=0, 1 of 48 calls came back valid and half were byte-identical to the failed answer; at T=0.4, 18 of 120 calls were valid and 7 of the 24 reviews were rescued by at least one of five calls. Temperature helps, but a single retry rescues a minority. [e4](results/e4_retry_temperature.md) |
| E5 | A numeric word limit is echoed and causes runaways | **Not reproduced** | "N palabras como máximo": 0 runaways in both runs and 2 echoes in 200 summaries (1 with the description in words). The numeric limit shortened the summaries (27.7 vs 33.9 words) and 133 of 200 still exceeded it. [e5](results/e5_summary_length.md) |
| E6 | Thinking and `format` do not mix: `think: false` | **Reproduced on the 4b** | qwen3:4b with `think` left out or `true`: **0 of 24** valid answers. The `response` came back empty and the JSON arrived in the `thinking` field (exactly the length and token count of the `think: false` answer for 13 of the 24 reviews; 1,003 vs 1,007 characters on average). With `think: false`, 24 of 24. qwen3:8b never produced thinking text with `format` set: 23 of 24 valid with `false`, 21-22 of 24 with the other two (3 runaways instead of 1). [e6](results/e6_thinking.md) |
| E7 | `localhost` costs time on every call on Windows | **Reproduced** | 50 calls in series: 2.1 s of client-side overhead per call with `localhost` vs 0.03 s with `127.0.0.1`; 9.9 vs 12.5 reviews/min. [e7](results/e7_localhost.md) |
| E8 | 4 threads with `OLLAMA_NUM_PARALLEL=4`; more threads do not help | **Partly** | qwen3:4b, two passes: 1 thread 12.9 / 12.0, 2 threads 10.6 / 11.4, 4 threads 21.0 / 21.7, 8 threads 25.2 / 24.0 reviews/min. Four threads nearly double the throughput, as in production, but here 8 still add 11-20 % (the extra requests queue in the server; median latency per review 10-11 s → 17-18 s) and 2 threads are slower than 1. qwen3:8b: 7.1, 8.9, 13.1, 15.2. [e8](results/e8_threads.md) |
| E9 | Without `keep_alive` the model is reloaded after 5 idle minutes | **Reproduced (one clean repetition)** | After 330 s idle with the server default, the model had been unloaded and the next call took 11.4 s (6.1 s of loading), against 5.0-5.2 s with `keep_alive: 30m`. The second repetition of the default case measures something else: it followed a request with `30m`, and a request that leaves `keep_alive` out did not shorten that timer (model still loaded, 5.1 s). [e9](results/e9_keep_alive.md) |
| E10 | Temperature 0 is not fully repeatable | **Partly** | One request at a time: 99 of 100 answers byte-identical. With 4 threads: 36 % byte-identical JSON, 80 % identical opinion sets, 42 % identical summaries; the flags stay at 98-100 %. The variation comes from running requests concurrently, not from temperature 0 itself. [e10](results/e10_repeatability.md) |
| L10 | Deterministic work belongs in code, thresholds in an empty gap | **Reproduced** | Of the 1,534 non-filler quotes in E1, none has a support in [0.4, 0.6); the 7 below that gap copy the topic definitions of the prompt. The post-processing removes both problems in the stored rows (0 repeated pairs; 99.2 % of the kept quotes literal). [quotes](results/quote_support.md), [gates](results/gates.md) |
| L11 | Do not ask the model what a detector knows | **Reproduced** | On the 38 reviews the detector marks as not Spanish, the model answered "es" for 37. Many of them are bilingual, but 11 contain no Spanish function word at all, and the model called all 11 Spanish. [l11](results/l11_language.md) |
| L12 | Alerts extracted, not inferred ("only when the text says it") | **Not reproduced: open problem** | The gate on 4-5 star reviews passes (0 alerts without a phrase), but the no-return alert fires on 40 of 40 one-star reviews with qwen3:4b (35 of 40 with the 8b), and 60 of its 85 hits have no no-return phrase that the deliberately broad regex can find. It looks as if the model infers the alert from the stars and the tone, whatever the wording says. A stronger model read the 85 hits and found the statement in 28; on the 100 reviews it labelled blind, the 4b raises the alert on all 20 one-star reviews where 5 say it (precision 0.36, recall 1.00). "Explicitly recommends" is over-flagged the same way (36 flagged, 6 explicit). [e1](results/e1_model_choice.md), [judge](results/judge_agreement.md) |
| L13 | Rewording the prompt does not move fine judgements | **Partly** | Incident field, 120 reviews: without the "none" default, recall on 1-2 stars went 40 % → 48 %; with example phrases, 31 %; as a list, 85 %, but then it fired 40 times on reviews with no incident phrase (11 with the production wording). Unlike production, the list never fired on 5-star reviews here. [incident](results/nice_incident_wording.md) |
| L14 | Gates on the real output catch what the schema cannot | **Mechanism only** | All gates pass on the 200-row pipeline run; `legal_action` never varied, but only 2 of those reviews contain a legal phrase, below the minimum to demand variation. [gates](results/gates.md) |
| L16 | Sometimes the stars are wrong, not the model | **Observed** | 4 of the 200 reviews have 4-5 stars and are read as negative by both layers. [gap](results/star_text_gap.md) |
| L17 | Business-reply judgement and its null anchoring | **Not reproducible here** | The dataset has no business replies. |
| Nice | `maxLength` on quotes and summary | **Not reproduced** | Limits at the 95th percentile of the E1 lengths: 21.6 vs 22.3 reviews/min (no gain; there were no runaways to cut); identical quotes 63 % vs 74 % for a plain repeat. [maxLength](results/nice_maxlength.md) |
| Nice | Instructions first, so the server caches the prefix | **Reproduced** | Prompt processing 0.22 s vs 0.55 s per call, but the answers change: identical opinion sets in 16 % of reviews (80 % for a repeat), no-return alerts 41 vs 85. Not adopted, as in production. 4 calls of this run got an HTTP 500 in the same second and are counted apart. [order](results/nice_prompt_order.md) |
| Nice | Skipping short positive reviews | **Reproduced** | 4-5 star reviews under 241 characters (21.8 % of the dataset): no alerts, but 11 of 22 name a staff member. [short](results/nice_short_positives.md) |

### What these numbers are not

- **No human accuracy figure.** The only reference labels come from a stronger model
  (Claude Opus 5.5), which labelled the 100 reviews of the `label.py` queue blind and read
  every alert the pipeline raised: [results/judge_agreement.md](results/judge_agreement.md).
  That is agreement with another model, not accuracy. The human gold set (`label.py`,
  `alerts_review.py`) is still pending, and `gold/` is empty.
- **The regex in `cues.py` is a broad net** used to find suspicious hits, never a label.
- **Speed numbers are from one laptop GPU** that throttles under long runs (the GPU
  temperature and clock are recorded before and after each run). Compare runs inside
  the same table, and treat differences under ~15 % as noise.
- **Balanced sample.** 40 reviews per star; per-star figures are more meaningful than
  totals.

## Output contract

`src/review_llm_eval/contract.py` (JSON Schema) and `prompt.py` (wording), written from
scratch for hotels. Every key is required; the model writes short keys and codes, and
`postprocess.expand` turns them into the column names of the database.

| Key | Column | Type | Content |
| --- | --- | --- | --- |
| `ops` | `opinions` | list of `{t, p, lit}` | one item per topic with a real opinion: topic code (13 topics: room, cleanliness, food, drinks, pools, beach, staff service, entertainment, value, front desk, noise, safety/health, grounds/location), polarity `pos` / `neg` / `neu`, and a short quote copied from the review |
| `emp` | `staff` | list of strings | names of hotel workers written in the review |
| `inc` | `incident` | enum, default `none` | long wait, room change, unexpected charge, booking problem, complaint ignored |
| `nov` `leg` `rob` `enf` `fra` | alerts | boolean | says they will not return / legal or formal action / theft / illness / accuses the hotel of bad faith; only when the text says it |
| `ret` `kid` `recom` | qualifiers | boolean | returning guest / travelling with children / explicitly recommends |
| `noc` | `nights` | integer or null | nights of the stay, when stated (nullable by type) |
| `idi` | raw output only | enum | language according to the model; the stored language comes from the detector |
| `rsm` | `summary` | string | "un par de frases" for 1-3 stars, "una frase breve" for 4-5 |

Stored next to them: layer 1 (`sentiment`, `sentiment_score`, `emotion`, `language`),
`raw_output` (the JSON as the model returned it), `llm_attempts`, `content_hash` and
`model_version` = `pysentimiento-robertuito+langdet-v1/v1+qwen3:4b/prompt-v1`.
[`results/example_outputs.jsonl`](results/example_outputs.jsonl) shows ten stored rows,
pseudonymised.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev,layer1]"
ollama pull qwen3:4b && ollama pull qwen3:8b         # compare the digests with config.py

python -m review_llm_eval.fetch                      # parquet -> data/raw/
python -m review_llm_eval.sample                     # data/sample.jsonl (200 reviews)
python -m review_llm_eval.ingest                     # SQLite store
python -m review_llm_eval.enrich --sample data/sample.jsonl   # both layers + gates

python -m review_llm_eval.experiments list
python -m review_llm_eval.experiments run E1         # E1..E10, NICE, L11, or a run id
python -m review_llm_eval.experiments run E4         # retries (after E1, E5 and E10)
python -m review_llm_eval.experiments run E9         # waits > 5 min, four times
python -m review_llm_eval.experiments analyze all    # -> results/*.md, results/runs.csv
python -m review_llm_eval.examples                   # example_outputs.jsonl, star_text_gap.md

python -m review_llm_eval.label --labeller <name>    # human gold set (pending)
python -m review_llm_eval.alerts_review --reviewer <name>
python -m review_llm_eval.evaluate --gold <labels.jsonl>   # score E1 against a label file
pytest && ruff check . && ruff format --check .
```

Every run records the server settings (read from Ollama's `server.log` on Windows), the
model digest, the GPU state and any model another process had loaded at the time.

## Data source and licence

- Dataset: [`beltrewilton/punta-cana-spanish-reviews`](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews)
  on Hugging Face, **MIT licence** (as declared on the dataset card), dataset commit
  `e4404ae54c02c1a53b9b3a5822b1fc2860a05fb1`, downloaded through the parquet API on
  2026-09-23. 34,561 Spanish hotel reviews; 33,038 remain after removing empty and
  duplicated texts.
- The dataset is **not redistributed**: `data/` is git-ignored, including the raw model
  answers (they quote the reviews). The committed `results/` files are aggregates;
  `example_outputs.jsonl` holds ten pipeline rows referenced by row index, with quotes
  cut to 80 characters and names replaced by placeholders.
- Code: MIT licence, see [`LICENSE`](LICENSE).

## About

A rebuild on public data of a system I designed and run in production. It contains no
code, prompts, data or figures from that system: the decisions table states its choices
without numbers, and every number in this README was measured here. Built with
AI-assisted development.
