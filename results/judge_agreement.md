# Agreement with a stronger model (not a human gold set)

**Reference labels.** The 100 reviews of the `label.py` queue (main sample, 20 per star
rating, seed 42), labelled blind by Claude Opus 5.5 (Anthropic) on 25 Sep 2026. It saw only
the title and the text (no rating, hotel or model output) and followed the definitions of
`label.py` and the prompt. The same model read the 100 alerts that qwen3:4b raised in the
pipeline store and had to copy the literal phrase stating each one.

**What this is not.** Agreement with another model, not accuracy: a bias both models share
does not show here. The human gold set (`label.py`) is still pending. The labels stay local
(`data/judge/`, git-ignored) because they contain staff names and quotes from the reviews.

Commands:

```bash
python -m review_llm_eval.evaluate --gold data/judge/judge_labels.jsonl --results data/judge_eval
python -m review_llm_eval.alerts_review --verdicts data/judge/judge_alert_verdicts.jsonl --report --results data/judge_eval
```

## E1 runs against the reference labels (n = 100)

| | qwen3:4b | qwen3:8b |
| --- | --- | --- |
| Topics mentioned: precision / recall | 0.81 / 0.65 | 0.76 / 0.71 |
| (topic, polarity) pairs: precision / recall / F1 | 0.73 / 0.55 / **0.63** | 0.70 / 0.63 / **0.66** |
| Staff names: precision / recall | 0.88 / 0.79 | 0.93 / 0.84 |
| Incident detected (any code), of 38 in the reference | 31 flagged, 28 right | 37 flagged, 32 right |
| Incident with the same code as the reference | 26 of 38 | 26 of 38 |
| "Explicitly recommends", 6 in the reference | 36 flagged, all 6 found | 26 flagged, all 6 found |
| No-return alert, 15 in the reference: precision / recall | 0.36 / 1.00 | 0.45 / 1.00 |
| Bad-faith alert, 8 in the reference: precision / recall | 1.00 / 0.50 | 0.86 / 0.75 |
| Illness (2) / theft (1) / legal action (1) found | 1 / 1 / 0 | 1 / 1 / 0 |

No-return alert by star rating (20 reviews per rating in this subset):

| Stars | qwen3:4b | qwen3:8b | Reference |
| --- | --- | --- | --- |
| 1 | 20 | 17 | 5 |
| 2 | 19 | 15 | 10 |
| 3 | 3 | 1 | 0 |
| 4-5 | 0 | 0 | 0 |

Layer 1 (pysentimiento) overall sentiment against the reference: 75 of 100.

## Alerts raised by qwen3:4b in the pipeline store (200 reviews), read one by one

| Alert | Raised | Stated in words |
| --- | --- | --- |
| says_no_return | 85 | 28 |
| bad_faith | 10 | 9 |
| theft | 3 | 3 |
| illness | 2 | 2 |
| legal_action | 0 | - |

The title counts as part of the review, as it does for the pipeline, which sends it to the
model; 11 of the confirmed alerts rest on the title alone.

## What it says

- **The model choice is a trade-off, not a quality win.** Against the reference, the 8b finds
  a few more opinions and names (pair F1 0.66 vs 0.63); the 4b is 1.6 times faster (E1) and
  more precise on topics and bad faith. For a backfill of hundreds of thousands of reviews,
  speed decides.
- **The no-return alert is the weak field of both models.** The 4b raised it on all 20
  one-star reviews of this subset, where the reference finds the statement in 5: the model
  infers it from the tone, as L12 suspected.
- **"Explicitly recommends" is over-flagged the same way.** The prompt says that speaking well
  is not enough; the 4b still marks 36 reviews, and the reference finds an explicit
  recommendation in 6.
- **Incidents behave as designed:** few false positives (3 of 31 with the 4b), some misses
  (10 of 38).
