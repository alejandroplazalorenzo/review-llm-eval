# review-llm-eval

Classifies Spanish hotel reviews with a **local LLM** (Ollama) under a **JSON Schema**
(overall sentiment, 12 aspects with their own sentiment, a main complaint and a
"would return" flag), and then **measures how far that output can be trusted**:
schema validity, latency, agreement with star ratings, agreement between two models,
run-to-run stability, and the tooling for a human-labelled evaluation.

## Why

A classifier that is right 90 % of the time and one that is right 60 % of the time
produce equally well-formed JSON. Schema validity tells you the output can be parsed,
not that it is true. LLM output is only useful downstream (dashboards, alerts, reports)
if you know how often it is wrong and in which way. This repository keeps those two
questions apart and labels every number for what it is: a measurement of format, a
noisy proxy, an agreement between models, or (pending) accuracy against a human.

## Architecture

```
Hugging Face parquet (34,561 reviews)
        │  fetch.py
        ▼
data/raw/ ──► sample.py: de-duplicate, 70 reviews per star rating, seed 42 ──► data/sample.jsonl (350)
                                                                                     │
        ┌────────────────────────────────────────────────────────────────────────────┘
        ▼
run.py (4 worker threads) ── per review ─────────────────────────────────────────────┐
  prompt.py   system prompt + "Title / Review" (no stars, no hotel name)             │
  client.py   POST 127.0.0.1:11434/api/chat  format=<JSON Schema>, T=0, seed=42       │
  schema.py   jsonschema validation ── invalid? ──► 1 retry with the validator error  │
  classify.py record raw text, parsed output, attempts, latency, token counts         │
        ▼                                                                            │
data/outputs/<model>.jsonl (+ .run.json with wall-clock time) ◄───────────────────────┘
        │
        ├─► analyze.py      ──► results/*.csv, results/summary.md   (reliability, proxy, agreement)
        ├─► experiments.py  ──► results/stability.csv, results/optional_fields.csv
        └─► label.py (human, blind) ──► gold/gold.jsonl ──► evaluate.py ──► results/human_eval_*.csv
```

Everything is plain Python (`requests`, `jsonschema`, `pyarrow`); metrics are
implemented by hand in `metrics.py` and unit-tested against known values.

## Output contract

Schema: `src/review_llm_eval/schema.py` (version 1.0). Every field is **required**;
"unknown" is an explicit `null`. `additionalProperties: false` everywhere.

| Field | Type | Values | Meaning |
| --- | --- | --- | --- |
| `aspects` | object with 12 required keys | each `positive` \| `neutral` \| `negative` \| `mixed` \| `null` | opinion about `staff`, `cleanliness`, `room`, `food`, `pool_beach`, `location`, `value`, `check_in`, `noise`, `safety`, `entertainment`, `other`; `null` = not mentioned |
| `complaint` | string (≤ 120 chars) or `null` | free text | main complaint, in English, ≤ 15 words |
| `would_return` | boolean or `null` | `true` / `false` / `null` | only if the reviewer says so; `null` = not stated |
| `sentiment` | string | `positive` \| `neutral` \| `negative` \| `mixed` | overall sentiment of the review |

Keys are generated in this order on purpose (evidence first, verdict last). Every
stored result also keeps the raw model text, the number of attempts, the validator
error of each failed attempt, latency and Ollama's token counts.

## Design decisions

| Decision | Rejected alternative | Why |
| --- | --- | --- |
| **Local model** (Ollama, quantised 4-7B) | Hosted API | Reviews are customer data; locally nothing leaves the machine and a full re-run of the sample costs nothing (7-8 minutes here), so every prompt change can be re-measured. The Ollama digest pins the exact weights. The price is smaller models with the failure modes described below. |
| **Constrained decoding**: the schema goes in Ollama's `format` field | "Answer in JSON" in the prompt + parse/repair | 700 of 700 answers were schema-valid at the first attempt (see results). Every answer is still re-validated with `jsonschema`, because a grammar does not stop generation at the token limit and not every schema keyword is necessarily enforced during decoding. The prompt-only alternative was not measured here. |
| **Required + nullable** fields | Optional fields | Measured (50 reviews per model): with `complaint` and `would_return` made optional, both models still emitted both keys in 100 % of answers, but the keys were moved after `sentiment` in 100 % of answers. "Optional" did not mean "omitted when unknown"; it silently changed the generation order. Required + `null` keeps "not stated" explicit and the order under control. |
| **Fixed aspect keys** (all 12 always present) | A list of `{aspect, sentiment}` for mentioned aspects only | No duplicates, no invented aspect names, and every aspect gets an explicit decision, which makes comparison trivial. Observed cost: `gemma3:4b` fills aspects that are not discussed with `neutral` (see error analysis). The list design was not measured. |
| **Temperature 0 + fixed seed** | Sampling with self-consistency voting | Lowest variance at 1× cost. Measured: this is **not** bit-for-bit reproducible with 4 concurrent requests (see stability). |
| **One retry that shows the model its invalid answer and the validator error** | Retry the identical request | At temperature 0 an identical request is likely to reproduce the same answer. Note: the retry path never triggered in the real runs (0 invalid answers); it is covered by mocked tests only. |
| **Star rating hidden** from the model and from the human labeller | Give the rating as context | The rating is the proxy label; showing it would make the proxy check meaningless and anchor the human. |
| **Balanced sample** (70 per star) | Proportional sample | The dataset is 77 % five-star (26,708 / 34,561); a proportional sample of 350 would contain about 14 one-star and 9 two-star reviews. Consequence: the numbers below are not the accuracy on the natural distribution. |
| **Proxy metrics + inter-model agreement now, human gold next** | LLM-generated "gold" labels / LLM-as-judge | Stars are free and independent of the models but noisy; kappa between models shows *where* they disagree, not who is right. Labels produced by an LLM cannot tell you how often an LLM is wrong, so the only accuracy figure this repository will report comes from human labels. |
| `127.0.0.1` instead of `localhost` | `localhost` | On the development machine a GET to `/api/version` took 2.07 s via `localhost` and 0.015 s via `127.0.0.1`. |

## Results

All numbers were produced on **2026-09-23** by the commands shown, on the sample of
350 reviews (70 per star rating). Setup: Windows 11, RTX 3070 8 GB, Ollama 0.34.2
(`OLLAMA_NUM_PARALLEL=4`, flash attention on, `q8_0` KV cache), Python 3.13.7,
`qwen2.5:7b-instruct` (Q4_K_M, digest `845dbda0ea48`) and `gemma3:4b` (Q4_K_M, digest
`a2af6cc3eb7f`), options `temperature=0, seed=42, num_ctx=4096, num_predict=512`,
4 concurrent requests, one model loaded at a time. Full tables: [`results/`](results/)
(`summary.md` is the generated overview).

```
python -m review_llm_eval.run --model qwen2.5:7b-instruct
python -m review_llm_eval.run --model gemma3:4b
python -m review_llm_eval.analyze
```

### 1. Reliability of the structured output

| Model | Schema-valid | Valid at 1st attempt | Retries | Failures | Token-limit stops | Latency mean / p50 / p95 | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:7b-instruct | 350/350 (100 %) | 350/350 | 0 | 0 | 0 | 4.65 / 4.47 / 5.17 s | 51.5 reviews/min |
| gemma3:4b | 350/350 (100 %) | 350/350 | 0 | 0 | 0 | 5.50 / 5.40 / 6.32 s | 43.5 reviews/min |

Latency is per review as seen by the client, with 4 requests in flight. The maximum
was about 20 s in both runs: in each run one stall hit the 4 requests in flight at
that moment (cause not determined). Mean prompt / output tokens: 548 / 123 (qwen),
541 / 138 (gemma). The 4B model was slower than the 7B one on this machine.

### 2. Overall sentiment vs star rating (noisy proxy, **not** accuracy)

1-2 stars → `negative`, 4-5 stars → `positive`; 280 reviews. A `mixed` or `neutral`
answer counts as a miss against this binary proxy.

| Model | Proxy agreement (accuracy) | Macro-F1 | F1 negative | F1 positive |
| --- | --- | --- | --- | --- |
| qwen2.5:7b-instruct | 0.836 | 0.898 | 0.953 | 0.843 |
| gemma3:4b | 0.896 | 0.926 | 0.958 | 0.893 |

Confusion matrices (rows: proxy, columns: model):

| qwen2.5:7b-instruct | positive | neutral | negative | mixed |
| --- | --- | --- | --- | --- |
| **negative** (1-2★, n=140) | 0 | 0 | 132 | 8 |
| **positive** (4-5★, n=140) | 102 | 0 | 5 | 33 |

| gemma3:4b | positive | neutral | negative | mixed |
| --- | --- | --- | --- | --- |
| **negative** (1-2★, n=140) | 0 | 0 | 138 | 2 |
| **positive** (4-5★, n=140) | 113 | 2 | 10 | 15 |

Where the models differ is the 4-star reviews: qwen called 32 of 70 `mixed`, gemma 15.
3-star reviews (no proxy label, n=70): qwen 40 negative / 27 mixed / 2 positive /
1 neutral; gemma 48 negative / 15 mixed / 4 positive / 3 neutral.
Per-star breakdown: `results/sentiment_by_rating.csv`.

### 3. Agreement between the two models (agreement, **not** accuracy)

Cohen's kappa on the 350 reviews. Two models can agree and both be wrong.

| Item | Kappa | Raw agreement |
| --- | --- | --- |
| Overall sentiment (4 classes) | 0.781 | 86.9 % |
| `would_return` (true / false / null) | 0.687 | 81.4 % |
| Complaint present (non-null) | 0.935 | 97.4 % |
| Aspect sentiment, when both models mark the aspect (988 pairs) | 0.726 | 83.5 % |

Per aspect: "mentioned" = non-null; "with opinion" = `positive`, `negative` or `mixed`
(i.e. ignoring `neutral`). Rates are qwen / gemma.

| Aspect | Mentioned rate | Kappa (mentioned) | With-opinion rate | Kappa (with opinion) |
| --- | --- | --- | --- | --- |
| staff | 71 % / 95 % | 0.22 | 64 % / 80 % | 0.57 |
| cleanliness | 30 % / 44 % | 0.68 | 27 % / 30 % | 0.85 |
| room | 36 % / 49 % | 0.68 | 33 % / 38 % | 0.73 |
| food | 51 % / 68 % | 0.66 | 47 % / 58 % | 0.76 |
| pool_beach | 19 % / 38 % | 0.53 | 18 % / 27 % | 0.73 |
| location | 7 % / 25 % | 0.29 | 5 % / 5 % | 0.55 |
| value | 27 % / 40 % | 0.51 | 25 % / 26 % | 0.61 |
| check_in | 14 % / 37 % | 0.40 | 13 % / 21 % | 0.62 |
| noise | 5 % / 23 % | 0.16 | 3 % / 4 % | 0.68 |
| safety | 9 % / 24 % | 0.29 | 8 % / 7 % | 0.83 |
| entertainment | 23 % / 36 % | 0.60 | 22 % / 20 % | 0.83 |
| other | 29 % / 11 % | 0.14 | 29 % / 9 % | 0.17 |

Most of the aspect disagreement is `neutral`: once it is ignored, kappa rises for every
aspect (noise 0.16 → 0.68, safety 0.29 → 0.83). The catch-all `other` stays near chance.

### 4. Field usage (descriptive diagnostics, not accuracy)

| | qwen2.5:7b-instruct | gemma3:4b |
| --- | --- | --- |
| `would_return` non-null | 81.1 % | 95.4 % |
| `would_return` non-null on the 259 reviews with **no** return cue* | 77.6 % | 94.2 % |
| `complaint` non-null | 73.4 % | 72.0 % |
| Complaints that look Spanish despite "in English"** | 46.3 % | 0.0 % |
| Aspects marked per review | 3.23 | 4.89 |
| Reviews with all 12 aspects non-null | 0.6 % | 2.9 % |

\* A review has a return cue if it matches `volv|vuelv|vuelta|regres|repet|repit|otra vez|de nuevo|nunca más`
(deliberately broad). Without any of them the review almost certainly does not say
whether the reviewer would return, so a non-null value there is inferred, not extracted.
\*\* Crude stop-word vote (`analyze.looks_spanish`); spot-checked on 14 complaints per model.
Both are heuristics for spotting problems, not labels.

### 5. Run-to-run stability (same settings, 50 reviews)

```
python -m review_llm_eval.run --model <model> --limit 50 --tag rerun
python -m review_llm_eval.experiments stability
```

| Model | Identical JSON | Same sentiment | Same `would_return` | All 12 aspects identical | Aspect cells identical | Same complaint text |
| --- | --- | --- | --- | --- | --- | --- |
| qwen2.5:7b-instruct | 60 % | 100 % | 100 % | 84 % | 98.2 % | 66 % |
| gemma3:4b | 56 % | 100 % | 100 % | 84 % | 98.7 % | 64 % |

Temperature 0 with a fixed seed was not reproducible under 4 concurrent requests; the
labels were very stable, the free-text complaint was not (qwen even switched language
between runs for the same review). A sequential (1 request at a time) comparison was
started but aborted because another process began using the same Ollama server, which
made both timings and model residency meaningless: **pending**.

### 6. Optional fields under constrained decoding (50 reviews)

```
python -m review_llm_eval.run --model <model> --limit 50 --tag optional --schema-variant optional
python -m review_llm_eval.experiments optional
```

With `complaint` and `would_return` removed from `required`: 100 % valid for both
models; both keys present in 100 % of answers; key order `aspects > sentiment >
complaint > would_return` in 100 % of answers (required keys first). `complaint`
non-null 74 % in both variants for both models; `would_return` non-null went from 74 %
to 86 % (qwen) and from 96 % to 98 % (gemma). Overall sentiment matched the required
variant in 49/50 (qwen) and 48/50 (gemma) reviews; with n = 50 this says nothing
reliable about field order.

### 7. Human-labelled evaluation

**Human-labelled evaluation: pending (0/200 reviews labelled).**

`label.py` and `evaluate.py` are built and tested but the gold file is empty on purpose:
the labels have to come from a person, not from a model. This is the next step, and the
only one that turns the proxies above into an accuracy (per-aspect precision / recall /
F1, sentiment accuracy, `would_return` accuracy).

## Error analysis

About 20 disagreements (model vs stars, and model vs model) were read one by one from
`data/analysis/disagreements.jsonl`. Review ids are row indices in the public parquet
file, so every case can be checked. Patterns actually seen:

1. **The stars are wrong more often than the models.** Several 4-star reviews are
   plainly negative: #11980 is titled *"LO PEOR."* ("the worst") and only describes
   failed dinner bookings; #21984 says *"Es una lástima que [...] no se tenga la cultura
   atención al cliente"*. Both models answer `negative`, and the proxy counts it as an
   error. Part of the proxy "error" rate is label noise.
2. **Where `mixed` begins is the main source of model disagreement.** In #13487 (4★)
   *"Lo califico muy bueno por un percanse [...] de todas maneras recomiendo el hotel"*
   qwen says `mixed`, gemma `positive`; in #5467 (1★), which lists *"Primero lo positivo
   [...] Lo negativo: 1. La música en la noche..."*, qwen says `mixed` against a
   negative proxy. The prompt's "neither dominates" is read differently by each model;
   a human definition (the gold set) has to settle it.
3. **Over-weighting one long complaint.** #23506 (4★) is titled *"Genial en todo, muy mal
   servicio en mesa del buffet"* and ends *"El resto genial, gran hotel"*; gemma answers
   `negative` because most of the text describes the buffet service.
4. **Blaming the hotel for a third party.** In #4296 (4★, *"Muy buen resort [...]
   realmente todo incluido"*) the complaint is about an excursion company's cameraman
   who never delivered a DVD; gemma rates the review `negative` and sets
   `would_return: false`, which the review never says.
5. **`would_return` is inferred, not extracted.** Besides #4296 and #23506 (gemma:
   `false`, nothing stated), both models set `true` on #13487 from *"recomiendo el
   hotel"* (recommending is not returning). The field diagnostic above quantifies it:
   94 % (gemma) and 78 % (qwen) non-null on reviews with no return cue at all.
6. **`neutral` as a filler for "not mentioned" (gemma).** #8142 is two sentences about
   value and food; gemma returns 12 non-null aspects, 9 of them `neutral` (noise,
   safety, location, ...); qwen marks only `value`. The same happens on #11980. This is
   why kappa for "mentioned" is low and kappa for "with opinion" is much higher.
7. **Aspect sentiment flattened.** #14721 (2★) praises front desk and butlers but calls
   the beach-buffet staff and kids-club staff rude; qwen marks `staff: positive`, gemma
   `mixed` (the better reading).
8. **Missed aspects.** #21984's title is *"Playas hermosas, la gente no es amable"*;
   neither model marks `pool_beach`.
9. **Instructions the schema cannot enforce.** 46 % of qwen's complaints are in Spanish
   (e.g. *"Música nocturna muy fuerte interrumpiendo el descanso"*, #5467); on the rerun
   the same review came back in English. A schema can force a string, not a language.
10. **Source truncation leaks into the output.** The dataset cuts long reviews at about
    790 characters (110 of the 350 sampled). #11080 ends *"- Los shows no son"*, and
    qwen's complaint is exactly *"Los shows no son"*.

## How to run

Requires Python ≥ 3.11 and [Ollama](https://ollama.com) for the classification step.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Linux/macOS: .venv/bin/python
# (then use the venv's python for the commands below)

python -m review_llm_eval.fetch                     # download the parquet into data/raw/
python -m review_llm_eval.sample                    # data/sample.jsonl, 350 reviews, seed 42

ollama pull qwen2.5:7b-instruct && ollama pull gemma3:4b
python -m review_llm_eval.run --model qwen2.5:7b-instruct    # ~7 min on an RTX 3070
python -m review_llm_eval.run --model gemma3:4b              # ~8 min
python -m review_llm_eval.analyze                            # results/*.csv, summary.md

# side experiments
python -m review_llm_eval.run --model gemma3:4b --limit 50 --tag rerun
python -m review_llm_eval.experiments stability
python -m review_llm_eval.run --model gemma3:4b --limit 50 --tag optional --schema-variant optional
python -m review_llm_eval.experiments optional

# human evaluation
python -m review_llm_eval.label --labeller <your-name>       # blind: no stars, no model output
python -m review_llm_eval.evaluate                           # scores both models against gold/
```

`run.py` resumes where it stopped (already processed reviews are skipped) unless
`--overwrite` is given. The Ollama URL is `http://127.0.0.1:11434` (`config.py`).

## Tests

```bash
pytest          # 80 tests, offline
ruff check . && ruff format --check .
```

The LLM client is replaced by a scripted fake; an autouse fixture blocks every socket
connection, so no test can reach Ollama or the internet. Covered: schema validation
(valid and invalid examples, optional variant), response parsing, the retry logic
(invalid → valid, invalid twice, transport errors), prompt building (stars and hotel
never reach the model), sampling determinism, metrics on toy data with known answers
(confusion matrix, precision / recall / F1, macro-F1, Cohen's kappa textbook example,
percentiles), the labelling CLI with scripted input, and the gold evaluation.
Fixtures are ten synthetic reviews (< 3 KB). Verified locally on Python 3.13.7; the
GitHub Actions workflow (`.github/workflows/ci.yml`) runs ruff and pytest on Python
3.11 and 3.12 and has not run yet because the repository has not been pushed.

## Limitations and next steps

- **No accuracy yet.** Until `gold/gold.jsonl` has 200 human labels, the only
  sentiment "accuracy" is agreement with star ratings, and there is none for aspects,
  `would_return` or `complaint`. Next: label the 200 queued reviews; ideally a second
  person labels a subset to get human-human kappa, the ceiling any model can reach.
- **Prompt v2** targeting what was observed: `would_return` only on an explicit
  statement, a stricter (or no) `neutral` for aspects, an explicit rule for `mixed`,
  and the complaint language. It should be judged against the gold set, not against
  v1 or the other model.
- **Narrow data**: one destination (all-inclusive resorts in Punta Cana), Spanish only,
  350 reviews balanced by stars (not the natural 77 % five-star distribution), texts cut
  at ~790 characters by the source.
- **Two small quantised models, one prompt version, one run each** (plus 50-review
  reruns). Differences of a few points between the models are within what one would
  expect from a different sample.
- **Determinism**: sequential (non-concurrent) reproducibility is pending.
- **Timings** come from a shared desktop GPU; one ~15 s stall per run is unexplained.
- The retry path is only exercised by tests; the real runs never produced invalid JSON.
- `other` is too vague to be useful (kappa 0.14-0.17); consider dropping it.

## Data source and licence

- Dataset: [`beltrewilton/punta-cana-spanish-reviews`](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews)
  on Hugging Face, **MIT licence** (as declared on the dataset card), dataset commit
  `e4404ae54c02c1a53b9b3a5822b1fc2860a05fb1`. 34,561 Spanish hotel reviews (hotel name,
  reviewer location, month, rating 1-5, title, text); 33,038 remain after removing
  empty and duplicated texts. Downloaded through the Hugging Face parquet API on
  2026-09-23.
- The dataset is **not redistributed** here: `data/` is git-ignored. The committed
  `results/` files are aggregates; `results/example_outputs.jsonl` holds 10 model
  outputs referenced by row index, without review text, hotel or reviewer data. The
  error analysis quotes short snippets for illustration.
- Code: MIT licence, see [`LICENSE`](LICENSE).

## Repository layout

```
src/review_llm_eval/
  config.py       taxonomy, label sets, defaults, paths
  fetch.py        download the parquet (Hugging Face API)
  data.py         loading, de-duplication, stratified sampling, gold queue
  sample.py       CLI: build data/sample.jsonl
  schema.py       output contract (JSON Schema) + validation
  prompt.py       system prompt, review formatting, retry message
  client.py       Ollama /api/chat client (structured outputs)
  classify.py     one review: call, validate, retry once, record
  run.py          CLI: batch run with worker threads
  metrics.py      confusion matrix, P/R/F1, macro-F1, Cohen's kappa, percentiles
  analyze.py      CLI: reliability, proxy, agreement, diagnostics -> results/
  experiments.py  CLI: stability and optional-field experiments
  label.py        CLI: blind human labelling -> gold/gold.jsonl
  evaluate.py     CLI: per-aspect P/R/F1 and sentiment accuracy vs gold
tests/            offline pytest suite (mocked LLM, synthetic fixtures)
results/          committed aggregate results (CSV / Markdown)
gold/             human labels (empty until labelled)
```

## About

Rebuild on public data of a system I designed and ran in production at work
(automotive sector, customer reviews). It contains no proprietary code, prompts or
data. Built with AI-assisted development; design decisions, evaluation and review are
mine.
