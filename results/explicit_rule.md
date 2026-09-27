# Explicit-only flags decided in code (L12)

**Problem.** The model raised "says no return" on almost every one-star review and "explicitly
recommends" on reviews that only praise the hotel, although the prompt asks for a statement
in words ([judge_agreement.md](judge_agreement.md), L12 in the README). Rewording the prompt
does not move judgements like these (L13).

**Rule** ([`rules.py`](../src/review_llm_eval/rules.py), `explicit-v1`). After the model, the
pipeline keeps a `True` only when the review (title and text) contains a phrase that could
state it:
- *says no return*: the broad no-return cue of `cues.py`;
- *explicitly recommends*: a form of "recomendar" not negated in the same clause ("no lo
  recomiendo" is a no-return statement; "no dudaría en recomendarlo" is a recommendation).

The rule only removes flags; it never adds one. The cues were written for the E1 checks
before the qwen3:8b run and **were not tuned on the reference labels below**.

**Cost.** No model call: the pipeline store keeps every raw answer, and
`enrich --remap` re-applies the post-processing to all 200 rows in seconds.

Commands:

```bash
python -m review_llm_eval.evaluate --gold data/judge/judge_labels.jsonl --results data/judge_eval --runs e1_4b e1_8b
python -m review_llm_eval.evaluate --gold data/judge/judge_labels.jsonl --results data/judge_eval_rules --runs e1_4b e1_8b --rules
python -m review_llm_eval.enrich --sample data/sample.jsonl --remap --only-layer1
python -m review_llm_eval.alerts_review --verdicts data/judge/judge_alert_verdicts.jsonl
```

## Against the stronger model's labels (E1 runs, 100 reviews)

| | qwen3:4b | qwen3:4b + rule | qwen3:8b | qwen3:8b + rule |
| --- | --- | --- | --- | --- |
| No-return: flagged / right (15 in the reference) | 42 / 15 | **13 / 12** | 33 / 15 | **13 / 12** |
| No-return: precision / recall | 0.36 / 1.00 | **0.92 / 0.80** | 0.45 / 1.00 | **0.92 / 0.80** |
| No-return raised on 1-star / 2-star reviews (20 each) | 20 / 19 | 5 / 8 | 17 / 15 | 5 / 8 |
| "Explicitly recommends": flagged / right (6 in the reference) | 36 / 6 | **7 / 6** | 26 / 6 | **6 / 6** |
| "Explicitly recommends": accuracy | 0.70 | **0.99** | 0.80 | **1.00** |

Every other field (opinions, staff names, incident, the other alerts) is unchanged.

## Alerts in the pipeline store (qwen3:4b, 200 reviews)

| | Before the rule | After the rule |
| --- | --- | --- |
| "Says no return" raised | 85 | 25 |
| Of those, stated in words (literal phrase found) | 28 | 23 |

The quality gates stay green after the re-map ([gates.md](gates.md)).

## What the rule gets wrong

- **Three true statements dropped**, the same three with both models. Each is phrased in a
  way the cue list does not cover:
  - "No vayan nunca" (advising against the hotel without "recomendar");
  - "No caigan en esta trampa" (about the hotel's membership sales, arguably not the hotel
    itself);
  - "trataré de no visitarlos nuevamente" ("visitar" is not in the no-return cue).
- **The cue list was not extended with those phrases.** That would tune the rule on the same
  labels it is scored against. Extending it needs reviews outside this set.
- **What remains is a precision/recall trade-off, chosen on purpose:** an alert feeds a
  person's inbox, so a missed one costs less than a false one.
