"""Experiments E1-E10 (+ the optional ones) that re-measure, on public data, the decisions
of the production pipeline.

Usage::

    python -m review_llm_eval.experiments list
    python -m review_llm_eval.experiments run E1          # model runs (needs Ollama)
    python -m review_llm_eval.experiments run E4          # retries of the failures seen
    python -m review_llm_eval.experiments run E9          # keep_alive (waits > 5 min twice)
    python -m review_llm_eval.experiments analyze all     # -> results/*.md

Raw per-review outputs go to ``data/experiments/`` (git-ignored); only aggregates are
written to ``results/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_llm_eval import analysis, checks, env
from review_llm_eval.client import LLMClient, OllamaClient
from review_llm_eval.config import (
    BATCH_SIZE,
    COMPARISON_MODEL,
    EXPERIMENTS_DIR,
    MAIN_MODEL,
    MODEL_NOTES,
    MODELS,
    RAW_PARQUET,
    RESULTS_DIR,
    SAMPLE_PATH,
    WORKERS,
)
from review_llm_eval.contract import ALERTS, BASE, Variant, build_schema
from review_llm_eval.cues import has_cue
from review_llm_eval.data import Review, load_parquet_rows, round_robin, rows_to_reviews
from review_llm_eval.extract import parse_and_validate, validator_for
from review_llm_eval.jsonl import append_jsonl, iter_jsonl, read_jsonl
from review_llm_eval.langdetect import detect
from review_llm_eval.langdetect import scores as language_scores
from review_llm_eval.metrics import mean, percentile
from review_llm_eval.postprocess import expand, hotel_veto, veto_for
from review_llm_eval.prompt import NUMERIC_LIMITS, build_prompt
from review_llm_eval.report import fmt, markdown_table
from review_llm_eval.runner import RunSpec, load_meta, load_records, prepare_server, run
from review_llm_eval.worker import run_in_batches

M4, M8 = MAIN_MODEL, COMPARISON_MODEL
FLAGS = analysis.FLAG_COLUMNS
SHORT = {M4: "4b", M8: "8b"}

LONG_KEYS = Variant(name="long_keys", keys="long")
NUMERIC = Variant(name="numeric_limit", summary="numeric")
FREE_OPTIONAL = Variant(name="free_text_optional", incident="free_text", incident_required=False)
FREE_REQUIRED = Variant(name="free_text_required", incident="free_text")
REORDER = Variant(name="instructions_first", order="instructions_first")
INC_NO_DEFAULT = Variant(name="incident_no_default", incident="no_default")
INC_PHRASES = Variant(name="incident_phrases", incident="phrases")
INC_LIST = Variant(name="incident_list", incident="list")


def _threads(model: str, suffix: str = "") -> list[RunSpec]:
    return [
        RunSpec(f"e8_{SHORT[model]}_t{w}{suffix}", "E8", model, "n48", workers=w)
        for w in (1, 2, 4, 8)
    ]


RUNS: dict[str, list[RunSpec]] = {
    "E1": [RunSpec("e1_4b", "E1", M4, "s200"), RunSpec("e1_8b", "E1", M8, "s200")],
    "E2": [RunSpec("e2_long_4b", "E2", M4, "s200", variant=LONG_KEYS)],
    "E3": [
        RunSpec(f"e3_{v.name}_{SHORT[m]}", "E3", m, "n100", variant=v)
        for m in (M4, M8)
        for v in (FREE_OPTIONAL, FREE_REQUIRED)
    ],
    "E5": [RunSpec("e5_numeric_4b", "E5", M4, "s200", variant=NUMERIC)],
    "E6": [
        RunSpec(f"e6_think_{label}_{SHORT[m]}", "E6", m, "n24", workers=1, think=think)
        for m in (M4, M8)
        for label, think in (("false", False), ("omitted", None), ("true", True))
    ],
    "E7": [
        RunSpec("e7_ip_4b", "E7", M4, "n50", workers=1),
        RunSpec("e7_localhost_4b", "E7", M4, "n50", workers=1, base_url="http://localhost:11434"),
    ],
    # The 4b curve is measured twice, at different times of the session (the laptop GPU
    # throttles under sustained load, so one pass alone can mislead).
    "E8": _threads(M4) + _threads(M8) + _threads(M4, "_b"),
    "E10": [
        RunSpec("e10_repeat_4b", "E10", M4, "s200"),
        RunSpec("e10_serial_a_4b", "E10", M4, "n100", workers=1),
        RunSpec("e10_serial_b_4b", "E10", M4, "n100", workers=1),
    ],
    "NICE": [
        RunSpec("nice_reorder_4b", "NICE", M4, "s200", variant=REORDER),
        RunSpec("nice_inc_no_default_4b", "NICE", M4, "n120", variant=INC_NO_DEFAULT),
        RunSpec("nice_inc_phrases_4b", "NICE", M4, "n120", variant=INC_PHRASES),
        RunSpec("nice_inc_list_4b", "NICE", M4, "n120", variant=INC_LIST),
    ],
    "L11": [RunSpec("l11_language_4b", "L11", M4, "foreign")],
}
"""E4 (retries) and E9 (keep_alive) are procedures, not plain runs; the maxLength run is
built from the E1 results (its limits are derived from the observed lengths)."""


# --- inputs ----------------------------------------------------------------------------


def load_sample(path: Path = SAMPLE_PATH) -> list[Review]:
    return [Review.from_dict(row) for row in read_jsonl(path)]


def subset(name: str, sample: Sequence[Review], parquet: Path = RAW_PARQUET) -> list[Review]:
    if name == "s200":
        return round_robin(sample)
    if name.startswith("n") and name[1:].isdigit():
        return round_robin(sample, int(name[1:]))
    if name == "foreign":
        reviews, _ = rows_to_reviews(load_parquet_rows(parquet))
        return [r for r in reviews if detect(r.full_text) != "es"]
    raise ValueError(f"unknown subset {name}")


def maxlength_spec(out_dir: Path) -> RunSpec:
    """maxLength run: limits = p95 of the lengths (characters) of the quotes and summaries
    the model wrote in the E1 4b run, so only the long tail is cut."""
    records = load_records(out_dir, "e1_4b")
    outputs = [o for r in records if (o := analysis.output_of(r))]
    quotes = [len(op["lit"]) for o in outputs for op in o["ops"]]
    summaries = [len(o["rsm"]) for o in outputs]
    if not quotes or not summaries:
        raise SystemExit("run E1 first: the maxLength limits are derived from e1_4b")
    limits = (int(percentile(quotes, 95)), int(percentile(summaries, 95)))
    variant = Variant(name="max_length", max_lengths=limits)
    return RunSpec("nice_maxlength_4b", "NICE", M4, "s200", variant=variant)


def all_specs(out_dir: Path) -> dict[str, RunSpec]:
    specs = {s.run_id: s for group in RUNS.values() for s in group}
    with contextlib.suppress(SystemExit):
        specs["nice_maxlength_4b"] = maxlength_spec(out_dir)
    return specs


# --- procedures (E4, E9): environment recorded like a plain run ------------------------


def _procedure_start(client: OllamaClient, model: str) -> dict[str, Any]:
    """Free the GPU of this project's other model and record the environment, in the
    same shape as a run session so the analysis prints it the same way."""
    unloaded, foreign = prepare_server(client, model)
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_done": 0,
        "environment": env.collect(client, model),
        "unloaded_before": unloaded,
        "foreign_models": foreign,
    }


def _procedure_end(
    out_dir: Path, name: str, session: dict[str, Any], client: OllamaClient, **spec: Any
) -> None:
    session["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    session["environment"]["gpu_state_after"] = env.gpu_state()
    loaded = [m["name"] for m in env.loaded_models(client)]
    session["foreign_models"] = sorted(
        set(session["foreign_models"]) | {n for n in loaded if n not in MODELS}
    )
    meta = load_meta(out_dir, name) or {"spec": spec, "sessions": []}
    meta["sessions"].append(session)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --- E4: retry temperature --------------------------------------------------------------

E4_SOURCES = (
    "e1_4b",
    "e10_repeat_4b",
    "e10_serial_a_4b",
    "e10_serial_b_4b",
    "e5_numeric_4b",
    "e1_8b",
)
"""Runs whose failed first attempts are retried (grouped by model: fewer reloads)."""
E4_T0_DRAWS = 2
E4_T04_DRAWS = 5
E4_INDUCED_MARGIN = 1.2


def _draws(
    client: LLMClient, model: str, prompt: str, schema: dict[str, Any], first_raw: str
) -> list[dict[str, Any]]:
    validator = validator_for(schema)
    draws = []
    for temperature, n in ((0.0, E4_T0_DRAWS), (0.4, E4_T04_DRAWS)):
        for _ in range(n):
            start = time.perf_counter()
            gen = client.generate(model, prompt, schema, temperature)
            obj, error = parse_and_validate(gen.response, validator)
            draws.append(
                {
                    "temperature": temperature,
                    "valid": obj is not None,
                    "error": error,
                    "done_reason": gen.done_reason,
                    "eval_count": gen.eval_count,
                    "same_as_first": gen.response == first_raw,
                    "latency_s": round(time.perf_counter() - start, 3),
                }
            )
    return draws


def induced_setup(out_dir: Path, sample: Sequence[Review]) -> tuple[int, list[Review]]:
    """Failures induced the way they happen in production (generation cut at
    ``num_predict``, JSON left unclosed): ``num_predict`` is lowered to the 75th percentile
    of the output tokens qwen3:4b wrote in E1, and the reviews are those whose E1 answer
    was longer than that by at most 20 %."""
    records = load_records(out_dir, "e1_4b")
    tokens = {
        r["review_id"]: r["extraction"]["attempts"][0]["eval_count"]
        for r in records
        if r.get("extraction")
    }
    if not tokens:
        raise SystemExit("run E1 first: the induced E4 is derived from e1_4b")
    cap = int(percentile(list(tokens.values()), 75))
    chosen = [r for r in sample if cap < tokens.get(r.review_id, 0) <= E4_INDUCED_MARGIN * cap]
    return cap, chosen


def run_e4_induced(
    sample: Sequence[Review], out_dir: Path, client: LLMClient | None = None
) -> None:
    """``client`` is for the tests; left out, the real server is used and its environment
    is recorded in ``e4_induced.meta.json``."""
    cap, reviews = induced_setup(out_dir, sample)
    real: OllamaClient | None = None
    session: dict[str, Any] | None = None
    if client is None:
        real = OllamaClient(num_predict=cap)
        session = _procedure_start(real, M4)
        client = real
    llm: LLMClient = client
    schema = build_schema(BASE)
    path = out_dir / "e4_induced.jsonl"
    done = {r["review_id"] for r in iter_jsonl(path)} if path.exists() else set()
    print(f"E4 induced: num_predict={cap}, {len(reviews)} reviews")

    def one(review: Review) -> dict[str, Any]:
        prompt = build_prompt(review, BASE)
        start = time.perf_counter()
        first = llm.generate(M4, prompt, schema, 0.0)
        obj, error = parse_and_validate(first.response, validator_for(schema))
        record: dict[str, Any] = {
            "review_id": review.review_id,
            "num_predict": cap,
            "first_valid": obj is not None,
            "first_error": error,
            "first_done_reason": first.done_reason,
            "first_eval_count": first.eval_count,
            "first_latency_s": round(time.perf_counter() - start, 3),
            "draws": [],
        }
        if obj is None:
            record["draws"] = _draws(llm, M4, prompt, schema, first.response)
        return record

    def handle(record: dict[str, Any]) -> bool:
        append_jsonl(path, record)
        if session is not None:
            session["n_done"] += 1
        print(f"E4 induced review {record['review_id']}: first valid={record['first_valid']}")
        return True

    todo = [r for r in reviews if r.review_id not in done]
    run_in_batches(todo, one, WORKERS, BATCH_SIZE, handle, lambda _: None)
    if real is not None and session is not None:
        _procedure_end(out_dir, "e4_induced", session, real, num_predict=cap, workers=WORKERS)


def run_e4(sample: Sequence[Review], specs: dict[str, RunSpec], out_dir: Path) -> None:
    """For every first attempt that failed in the source runs, call the model again with
    the same prompt: ``E4_T0_DRAWS`` times at temperature 0 and ``E4_T04_DRAWS`` at 0.4."""
    by_id = {r.review_id: r for r in sample}
    path = out_dir / "e4_retry.jsonl"
    done = {(r["source"], r["review_id"]) for r in iter_jsonl(path)} if path.exists() else set()
    for source in E4_SOURCES:
        spec = specs[source]
        pending = [
            rec
            for rec in load_records(out_dir, source)
            if ((rec.get("extraction") or {}).get("attempts") or [{"error": None}])[0]["error"]
            and (source, rec["review_id"]) not in done
        ]
        if not pending:
            print(f"E4 {source}: no failed first attempt to retry")
            continue
        client = spec.client()
        session = _procedure_start(client, spec.model)
        schema = build_schema(spec.variant)
        for rec in pending:
            first = rec["extraction"]["attempts"][0]
            review = by_id[rec["review_id"]]
            prompt = build_prompt(review, spec.variant)
            draws = _draws(client, spec.model, prompt, schema, first["raw"])
            append_jsonl(
                path,
                {
                    "source": source,
                    "model": spec.model,
                    "variant": spec.variant.name,
                    "review_id": review.review_id,
                    "first_error": first["error"],
                    "first_done_reason": first.get("done_reason"),
                    "draws": draws,
                },
            )
            session["n_done"] += 1
            print(f"E4 {source} review {review.review_id}: {sum(d['valid'] for d in draws)} valid")
        _procedure_end(out_dir, "e4_retry", session, client, sources=list(E4_SOURCES))


# --- E9: keep_alive -----------------------------------------------------------------------

E9_IDLE_S = 330  # server default keep_alive is 5 min
E9_REPS = 2


def run_e9(
    sample: Sequence[Review], out_dir: Path, sleep: Callable[[float], None] = time.sleep
) -> None:
    """Latency of the first call after more than 5 minutes idle, with the request's
    ``keep_alive`` left out (server default, 5 min) and set to "30m". The two alternate,
    so from the second rep on the "omitted" call follows a "30m" request; the analysis
    reports whether the model was still loaded before each timed call."""
    review = round_robin(sample, 1)[0]
    prompt, schema = build_prompt(review, BASE), build_schema(BASE)
    path = out_dir / "e9_keep_alive.jsonl"
    server = OllamaClient()
    session = _procedure_start(server, M4)
    print(
        f"E9: unloaded {session['unloaded_before'] or 'nothing'}; "
        f"other users' models loaded: {session['foreign_models'] or 'none'}"
    )
    for rep in range(E9_REPS):
        for label, keep in (("omitted", None), ("30m", "30m")):
            client = OllamaClient(keep_alive=keep)
            client.generate(M4, prompt, schema, 0)  # load / refresh the timer
            sleep(E9_IDLE_S)
            loaded = [m["name"] for m in env.loaded_models(client)]
            start = time.perf_counter()
            gen = client.generate(M4, prompt, schema, 0)
            append_jsonl(
                path,
                {
                    "rep": rep,
                    "keep_alive": label,
                    "idle_s": E9_IDLE_S,
                    "loaded_before_call": loaded,
                    "latency_s": round(time.perf_counter() - start, 3),
                    "load_duration_s": gen.load_duration_s,
                    "total_duration_s": gen.total_duration_s,
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            session["n_done"] += 1
            print(f"E9 rep {rep} keep_alive={label}: loaded={loaded}")
    _procedure_end(out_dir, "e9_keep_alive", session, server, idle_s=E9_IDLE_S, reps=E9_REPS)


# --- analysis -----------------------------------------------------------------------------


def _session(out_dir: Path, run_id: str) -> dict[str, Any] | None:
    sessions = load_meta(out_dir, run_id).get("sessions") or []
    return sessions[-1] if sessions else None


def _env_block(out_dir: Path, run_ids: Sequence[str], command: str) -> list[str]:
    lines = ["## Setup", "", f"Command: `{command}`", ""]
    seen: set[str] = set()
    for run_id in run_ids:
        session = _session(out_dir, run_id)
        if not session:
            lines.append(f"- `{run_id}`: not run (pending)")
            continue
        environment = session.get("environment") or {}
        model = environment.get("model") or {}
        server = environment.get("server") or {}
        foreign = session.get("foreign_models") or []
        lines.append(
            f"- `{run_id}`: {session['started_at'][:10]}, n={session['n_done']}, "
            f"{model.get('model')} digest `{model.get('digest')}`"
            + (f", **other models loaded: {', '.join(foreign)}**" if foreign else "")
        )
        key = json.dumps([environment.get("ollama_version"), server, environment.get("gpu")])
        if key not in seen:
            seen.add(key)
            lines.append(
                f"  Ollama {environment.get('ollama_version')}, GPU {environment.get('gpu')}, "
                + ", ".join(f"{k}={v}" for k, v in server.items() if k != "source")
                + f" (from {server.get('source')}), Python {environment.get('python')}"
            )
    return [*lines, ""]


def _write(results: Path, name: str, title: str, body: list[str]) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / name
    path.write_text("\n".join([f"# {title}", "", *body, ""]), encoding="utf-8")
    print(f"-> {path}")
    return path


def _summary_table(out_dir: Path, run_ids: Sequence[str]) -> list[str]:
    headers = [
        "run", "n", "valid", "valid 1st", "retried", "rescued", "failed", "net err",
        "runaway", "out tok mean", "out tok p95", "prompt tok", "prompt eval s",
        "latency p50 s", "decode tok/s", "reviews/min",
    ]  # fmt: skip
    rows = []
    for run_id in run_ids:
        records = load_records(out_dir, run_id)
        if not records:
            rows.append([run_id, "pending", *[""] * (len(headers) - 2)])
            continue
        s = analysis.run_summary(records, _session(out_dir, run_id))
        rows.append(
            [
                run_id, s["n"], s["valid"], s["valid_first_attempt"], s["retried"],
                s["rescued_by_retry"], s["failed"], s["network_errors"], s["runaway_reviews"],
                s["output_tokens_mean"], s["output_tokens_p95"], s["prompt_tokens_mean"],
                fmt(s["prompt_eval_s_mean"], 2), s["latency_p50_s"], s["decode_tok_per_s"],
                s["reviews_per_min"],
            ]
        )  # fmt: skip
    return [markdown_table(headers, rows, digits=1), ""]


def analyze_e1(out_dir: Path, results: Path, sample: list[Review]) -> None:
    reviews = {r.review_id: r for r in sample}
    a, b = load_records(out_dir, "e1_4b"), load_records(out_dir, "e1_8b")
    body = _env_block(out_dir, ["e1_4b", "e1_8b"], "python -m review_llm_eval.experiments run E1")
    body += [
        f"- qwen3:4b: {MODEL_NOTES[M4]}",
        f"- qwen3:8b: {MODEL_NOTES[M8]}",
        "",
        "Sample: 200 reviews, 40 per star rating; 4 client threads, "
        "OLLAMA_NUM_PARALLEL=4; production settings.",
        "",
        "## Reliability and speed",
        "",
        *_summary_table(out_dir, ["e1_4b", "e1_8b"]),
        "Thread curve (1/2/4/8 threads, 48 reviews): see `e8_threads.md`.",
        "",
    ]
    if a and b:
        ca, cb = checks.compute(a, reviews), checks.compute(b, reviews)
        verdicts = checks.compare(ca, cb)
        rows = [
            [c.name, c.description, ca[c.name], cb[c.name], {"a": "4b", "b": "8b"}.get(v, v)]
            for c, v in zip(checks.CHECKS, verdicts, strict=True)
        ]
        wins4 = sum(v in ("a", "tie") for v in verdicts)
        wins8 = sum(v in ("b", "tie") for v in verdicts)
        body += [
            "## The ten objective checks (`checks.py`, fixed before the qwen3:8b run)",
            "",
            markdown_table(["check", "what", "qwen3:4b", "qwen3:8b", "better"], rows, digits=3),
            "",
            f"qwen3:4b better or equal in **{wins4} of 10**; qwen3:8b better or equal in "
            f"**{wins8} of 10**.",
            "",
            f"Fields that never varied: 4b `{ca['fields_never_varying'] or '-'}`, "
            f"8b `{cb['fields_never_varying'] or '-'}`. Staff names (job titles excluded): "
            f"{ca['n_names']} / {cb['n_names']}, of which not in the review "
            f"{ca['n_names_missing']} / {cb['n_names_missing']}. Quotes: {ca['n_quotes']} / "
            f"{cb['n_quotes']}, of which fillers {ca['n_fillers']} / {cb['n_fillers']} and "
            f"unsupported {ca['n_unsupported']} / {cb['n_unsupported']}.",
            "",
        ]
        flag_rows = [
            [run_id, *(analysis.flag_counts(recs, BASE, reviews).get(c, 0) for c in FLAGS)]
            for run_id, recs in (("e1_4b", a), ("e1_8b", b))
        ]
        by_rating = [
            [run_id, *(f"{x}/{n} ({y})" for n, x, y in by.values())]
            for run_id, recs in (("e1_4b", a), ("e1_8b", b))
            if (by := analysis.alert_by_rating(recs, BASE, reviews, "says_no_return"))
        ]
        body += [
            "## How often each field fires (not one of the ten checks)",
            "",
            "Counts over valid outputs, after post-processing. `_no_phrase` = raised on a",
            "review that contains no phrase for that alert (`cues.py`, deliberately broad).",
            "",
            markdown_table(["run", *FLAGS], flag_rows),
            "",
            "`says_no_return` per star rating: raised / valid outputs (of those, raised on a",
            "review with no such phrase). The check above only looks at 5-star reviews.",
            "",
            markdown_table(["run", *(f"{n} star" for n in range(1, 6))], by_rating),
            "",
        ]
    _write(results, "e1_model_choice.md", "E1: qwen3:4b or qwen3:8b?", body)


def analyze_quote_support(out_dir: Path, results: Path, sample: list[Review]) -> None:
    reviews = {r.review_id: r for r in sample}
    body = _env_block(
        out_dir, ["e1_4b", "e1_8b"], "python -m review_llm_eval.experiments analyze all"
    )
    body += [
        "Support = share of a quote's content words (4+ letters) found in the review;",
        "1.0 = literal after folding case, accents and spacing. Fillers excluded.",
        "",
    ]
    rows: list[list[Any]] = []
    columns = []
    for run_id in ("e1_4b", "e1_8b"):
        pairs = [
            (str(op.get("lit") or ""), reviews[r["review_id"]].full_text)
            for r in load_records(out_dir, run_id)
            if (o := analysis.output_of(r))
            for op in o["ops"]
        ]
        columns.append(dict(analysis.support_histogram(pairs)))
    labels = list(columns[0]) if columns else []
    rows = [[label, *(c.get(label, 0) for c in columns)] for label in labels]
    body += [markdown_table(["support", "qwen3:4b", "qwen3:8b"], rows), ""]
    _write(results, "quote_support.md", "Quote support distribution (threshold derivation)", body)


def analyze_pairwise(
    out_dir: Path,
    results: Path,
    file_name: str,
    title: str,
    pairs: Sequence[tuple[str, str, Variant, Variant]],
    command: str,
    extra: list[str] | None = None,
    sample: Sequence[Review] | None = None,
) -> None:
    run_ids = sorted({x for a, b, _, _ in pairs for x in (a, b)})
    body = _env_block(out_dir, run_ids, command)
    body += ["## Reliability and speed", "", *_summary_table(out_dir, run_ids)]
    headers = ["comparison", "pairs", *analysis.FIELD_VIEWS]
    rows = []
    for a_id, b_id, va, vb in pairs:
        a, b = load_records(out_dir, a_id), load_records(out_dir, b_id)
        if not a or not b:
            rows.append([f"{a_id} vs {b_id}", "pending"])
            continue
        ag = analysis.agreement(a, b, va, vb)
        rows.append(
            [f"{a_id} vs {b_id}", ag["n_pairs"], *(ag.get(f) for f in analysis.FIELD_VIEWS)]
        )
    body += [
        "## Identical answers per field (share of reviews valid in both runs)",
        "",
        markdown_table(headers, rows, digits=2),
        "",
        *(extra or []),
        "",
    ]
    if sample is not None:
        by_id = {r.review_id: r for r in sample}
        variants = {a: va for a, _, va, _ in pairs} | {b: vb for _, b, _, vb in pairs}
        flag_rows = []
        for run_id in run_ids:
            records = load_records(out_dir, run_id)
            if records:
                counts = analysis.flag_counts(records, variants[run_id], by_id)
                flag_rows.append([run_id, *(counts.get(c, 0) for c in analysis.FLAG_COLUMNS)])
        body += [
            "## How often each field fires",
            "",
            "Counts over valid outputs, after post-processing. `_no_phrase` = raised on a review",
            "that contains no phrase for that alert (`cues.py`, deliberately broad).",
            "",
            markdown_table(["run", *analysis.FLAG_COLUMNS], flag_rows),
            "",
        ]
    _write(results, file_name, title, body)


def analyze_e3(out_dir: Path, results: Path) -> None:
    run_ids = [s.run_id for s in RUNS["E3"]]
    body = _env_block(out_dir, run_ids, "python -m review_llm_eval.experiments run E3")
    body += [
        "The incident field in the shape of the first production contract: free text, "
        '`["string", "null"]`. `free_text_optional` leaves it out of `required`; '
        "`free_text_required` keeps it in. 100 reviews (20 per star).",
        "",
    ]
    rows = []
    for spec in RUNS["E3"]:
        records = load_records(out_dir, spec.run_id)
        if not records:
            rows.append([spec.run_id, "pending"])
            continue
        p = analysis.key_presence(records, "inc")
        rows.append(
            [
                spec.run_id,
                p["n_valid"],
                p["key_present"],
                p["non_null_when_present"],
                p["non_null_overall"],
            ]
        )
    body += [
        markdown_table(
            ["run", "valid", "key present", "non-null when present", "non-null overall"], rows
        ),
        "",
        *_summary_table(out_dir, run_ids),
    ]
    _write(results, "e3_optional_fields.md", "E3: does an optional field get omitted?", body)


def analyze_e4(out_dir: Path, results: Path) -> None:
    path = out_dir / "e4_retry.jsonl"
    body = _env_block(
        out_dir,
        [*E4_SOURCES, "e4_induced", "e4_retry"],
        "python -m review_llm_eval.experiments run E4",
    )
    rows_in = list(iter_jsonl(path)) if path.exists() else []
    # What the production retry (one call at 0.4) did inside every run: the natural
    # failures, without inducing anything.
    observed: list[list[Any]] = []
    totals: dict[str, list[int]] = {}
    runaway_runs: dict[int, list[str]] = {}
    appearances: dict[int, int] = {}
    for run_id, spec in all_specs(out_dir).items():
        records = load_records(out_dir, run_id)
        firsts = [r for r in records if (r.get("extraction") or {}).get("attempts")]
        for r in firsts:
            if spec.model == M8:
                appearances[r["review_id"]] = appearances.get(r["review_id"], 0) + 1
        failed = [r for r in firsts if r["extraction"]["attempts"][0]["error"]]
        runaway = [
            r for r in failed if r["extraction"]["attempts"][0].get("done_reason") == "length"
        ]
        rescued = [r for r in runaway if r["extraction"]["output"] is not None]
        total = totals.setdefault(SHORT[spec.model], [0, 0, 0])
        total[0] += len(firsts)
        total[1] += len(runaway)
        total[2] += len(rescued)
        for r in runaway:
            runaway_runs.setdefault(r["review_id"], []).append(run_id)
        if failed or run_id in E4_SOURCES:
            observed.append(
                [run_id, len(records), len(failed), len(runaway), len(rescued),
                 len(failed) - len(runaway)]
            )  # fmt: skip
    repeat = ", ".join(
        f"review {rid}: {len(runs)} of {appearances.get(rid, len(runs))} qwen3:8b runs"
        for rid, runs in sorted(runaway_runs.items(), key=lambda x: -len(x[1]))
    )
    body += [
        "## Natural failures inside the runs (first attempt T=0, retry T=0.4, as in production)",
        "",
        "Every run of every experiment; listed when a first attempt failed (plus the E4",
        "sources). 'Other' failures are not cut at `num_predict`: in E6 they are the answers",
        "that came back in the `thinking` field.",
        "",
        markdown_table(
            ["run", "reviews", "1st attempt invalid", "cut at num_predict (runaway)",
             "runaways rescued by the retry", "other failures"],
            observed,
        ),
        "",
        markdown_table(
            ["model", "first attempts (all runs)", "runaways", "rescued by the T=0.4 retry"],
            [[m, *v] for m, v in sorted(totals.items())],
        ),
        "",
        f"Reviews that ran away: {repeat or 'none'}.",
        "",
    ]  # fmt: skip
    body += ["## Retries of the failures seen in those runs", ""]
    body += _draw_table(rows_in) if rows_in else ["No failed first attempt to retry.", ""]
    induced_path = out_dir / "e4_induced.jsonl"
    induced = list(iter_jsonl(induced_path)) if induced_path.exists() else []
    body += ["## Induced failures", ""]
    if induced:
        failed = [r for r in induced if not r["first_valid"]]
        body += [
            f"`num_predict` lowered to {induced[0]['num_predict']} (75th percentile of the "
            f"output tokens of qwen3:4b in E1) on the {len(induced)} reviews whose E1 answer "
            f"was up to {E4_INDUCED_MARGIN - 1:.0%} longer. First attempt (T=0) invalid: "
            f"{len(failed)} of {len(induced)} (cut at num_predict: "
            f"{sum(r['first_done_reason'] == 'length' for r in failed)}).",
            "",
            "These are complete answers cut short, not a loop like the natural runaways of",
            "qwen3:8b (an opinion list that repeats the same items until `num_predict`): a",
            "retry only succeeds if the model happens to write a shorter answer.",
            "",
            *(_draw_table(failed) if failed else []),
        ]
    else:
        body += ["Pending.", ""]
    _write(results, "e4_retry_temperature.md", "E4: which temperature for the retry?", body)


def _draw_table(rows_in: Sequence[dict[str, Any]]) -> list[str]:
    table = []
    for temperature in (0.0, 0.4):
        draws = [d for r in rows_in for d in r["draws"] if d["temperature"] == temperature]
        rescued = sum(
            any(d["valid"] for d in r["draws"] if d["temperature"] == temperature) for r in rows_in
        )
        table.append(
            [
                f"{temperature:g}",
                len(draws),
                sum(d["valid"] for d in draws),
                sum(d["valid"] for d in draws) / len(draws) if draws else float("nan"),
                f"{rescued}/{len(rows_in)}",
                sum(d["same_as_first"] for d in draws),
            ]
        )
    headers = [
        "retry temperature",
        "calls",
        "valid calls",
        "share of calls valid",
        "failures rescued by any call",
        "calls identical to the failed answer",
    ]
    return [
        markdown_table(headers, table),
        "",
        f"Draws per failure: {E4_T0_DRAWS} at T=0, {E4_T04_DRAWS} at T=0.4, same prompt.",
        "",
    ]


def analyze_e5(out_dir: Path, results: Path) -> None:
    run_ids = ["e1_4b", "e5_numeric_4b"]
    body = _env_block(out_dir, run_ids, "python -m review_llm_eval.experiments run E5")
    body += [
        f'`words`: summary length described in words (production). `numeric`: "un resumen de '
        f'N palabras como máximo" with N = {NUMERIC_LIMITS["long"]} for 1-3 stars and '
        f"{NUMERIC_LIMITS['short']} for 4-5. 'Over the numeric limit' applies the same N to",
        "both runs. Echo = the summary mentions words or a maximum.",
        "",
    ]

    def limit_for(rating: int | None) -> int:
        return NUMERIC_LIMITS["long" if rating is not None and rating <= 3 else "short"]

    rows = []
    for run_id, variant in (("e1_4b", BASE), ("e5_numeric_4b", NUMERIC)):
        records = load_records(out_dir, run_id)
        if not records:
            rows.append([run_id, "pending"])
            continue
        s = analysis.summary_stats(records, variant, limit_for)
        summary = analysis.run_summary(records, _session(out_dir, run_id))
        rows.append(
            [
                run_id, s["n"], s["valid"], s["runaway_first_attempt"],
                s["echo_in_valid_summaries"], s["echo_in_first_raw"], s["summary_words_mean"],
                s["summary_words_p95"], s["summary_words_max"], s["over_limit"],
                summary["output_tokens_mean"], summary["output_tokens_p95"],
                summary["output_tokens_max"],
            ]
        )  # fmt: skip
    body += [
        markdown_table(
            ["run", "n", "valid", "1st attempt cut at num_predict", "echo in valid summaries",
             "echo in 1st raw answer", "summary words mean", "summary words p95",
             "summary words max", "over the numeric limit", "out tok mean", "out tok p95",
             "out tok max"],
            rows,
        ),
        "",
    ]  # fmt: skip
    _write(results, "e5_summary_length.md", "E5: numeric word limit or a description?", body)


def analyze_e6(out_dir: Path, results: Path) -> None:
    run_ids = [s.run_id for s in RUNS["E6"]]
    body = _env_block(out_dir, run_ids, "python -m review_llm_eval.experiments run E6")
    body += ["24 reviews, 1 thread, `format` = schema in all runs; only `think` changes.", ""]
    rows = []
    for spec in RUNS["E6"]:
        records = load_records(out_dir, spec.run_id)
        if not records:
            rows.append([spec.run_id, "pending"])
            continue
        firsts = {
            r["review_id"]: r["extraction"]["attempts"][0] for r in records if r.get("extraction")
        }
        # The same review answered with think=false by the same model: is the "thinking"
        # the answer itself (same length, same token count)?
        plain = {
            r["review_id"]: r["extraction"]["attempts"][0]
            for r in load_records(out_dir, f"e6_think_false_{SHORT[spec.model]}")
            if r.get("extraction")
        }
        s = analysis.run_summary(records, _session(out_dir, spec.run_id))
        thinking = [a.get("thinking_chars") or 0 for a in firsts.values()]
        same = sum(
            bool(a.get("thinking_chars"))
            and rid in plain
            and a["eval_count"] == plain[rid]["eval_count"]
            and a["thinking_chars"] == len(plain[rid]["raw"])
            for rid, a in firsts.items()
        )
        rows.append(
            [
                spec.run_id, s["n"], s["valid_first_attempt"], s["valid"],
                s["runaway_reviews"], s["output_tokens_mean"],
                mean([a["latency_s"] for a in firsts.values()]),
                mean([len(a["raw"]) for a in firsts.values()]),
                sum(t > 0 for t in thinking), mean(thinking), same,
            ]
        )  # fmt: skip
    body += [
        markdown_table(
            ["run", "n", "valid 1st", "valid", "runaway", "out tok mean", "latency 1st s",
             "response chars mean", "answers with thinking", "thinking chars mean",
             "thinking identical in size to the think=false answer"],
            rows,
        ),
        "",
        "Last column: first attempts whose `thinking` field has exactly the length (in",
        "characters) and the token count of the same model's think=false answer to the same",
        "review, i.e. the JSON answer delivered in the `thinking` field.",
        "",
    ]  # fmt: skip
    _write(results, "e6_thinking.md", "E6: thinking and `format`", body)


def analyze_e7(out_dir: Path, results: Path) -> None:
    run_ids = ["e7_ip_4b", "e7_localhost_4b"]
    body = _env_block(out_dir, run_ids, "python -m review_llm_eval.experiments run E7")
    body += ["50 reviews, 1 thread, same everything except the host name in the URL.", ""]
    rows = []
    for run_id in run_ids:
        records = load_records(out_dir, run_id)
        if not records:
            rows.append([run_id, "pending"])
            continue
        attempts = [a for r in records for a in r["extraction"]["attempts"]] if records else []
        client = [a["latency_s"] for a in attempts]
        server = [a["total_duration_s"] for a in attempts if a.get("total_duration_s")]
        s = analysis.run_summary(records, _session(out_dir, run_id))
        rows.append(
            [
                run_id,
                len(attempts),
                mean(client),
                mean(server),
                mean(client) - mean(server),
                s["reviews_per_min"],
            ]
        )
    body += [
        markdown_table(
            ["run", "calls", "client s/call", "server s/call", "overhead s/call", "reviews/min"],
            rows,
        ),
        "",
    ]
    _write(results, "e7_localhost.md", "E7: localhost or 127.0.0.1?", body)


def analyze_e8(out_dir: Path, results: Path) -> None:
    run_ids = [s.run_id for s in RUNS["E8"]]
    body = _env_block(out_dir, run_ids, "python -m review_llm_eval.experiments run E8")
    body += [
        "48 reviews per run, server OLLAMA_NUM_PARALLEL=4, batches of 25 as in the pipeline.",
        "",
    ]
    body += _summary_table(out_dir, run_ids)
    _write(results, "e8_threads.md", "E8: how many client threads?", body)


def analyze_e9(out_dir: Path, results: Path) -> None:
    path = out_dir / "e9_keep_alive.jsonl"
    rows_in = list(iter_jsonl(path)) if path.exists() else []
    body = _env_block(out_dir, ["e9_keep_alive"], "python -m review_llm_eval.experiments run E9")
    body += [
        f"qwen3:4b, one review, 1 thread. Before each timed call the model is used once "
        f"and then left idle for {E9_IDLE_S} s (the server default keep_alive is 5 min).",
        "",
    ]
    if rows_in:
        body += [
            markdown_table(
                ["rep", "keep_alive", "model loaded before the call", "latency s", "load s", "at"],
                [
                    [r["rep"], r["keep_alive"], ", ".join(r["loaded_before_call"]) or "none",
                     r["latency_s"], r["load_duration_s"], r["at"]]
                    for r in rows_in
                ],
            ),
            "",
        ]  # fmt: skip
        still_loaded = [
            r["rep"] for r in rows_in if r["keep_alive"] == "omitted" and r["loaded_before_call"]
        ]
        if still_loaded:
            body += [
                f"Rep(s) {', '.join(map(str, still_loaded))}: the call without `keep_alive` came "
                "right after a request with `keep_alive: 30m`, and the model was still loaded "
                f"after {E9_IDLE_S} s. A request that leaves `keep_alive` out did not shorten the "
                "timer an earlier request had set, so only the other rep(s) of `omitted` measure "
                "a reload after the default 5 minutes.",
                "",
            ]
    else:
        body += ["Pending.", ""]
    _write(results, "e9_keep_alive.md", "E9: is keep_alive needed?", body)


def analyze_incident_wording(out_dir: Path, results: Path, sample: list[Review]) -> None:
    reviews = {r.review_id: r for r in sample}
    variants = [
        ("e1_4b", BASE),
        ("nice_inc_no_default_4b", INC_NO_DEFAULT),
        ("nice_inc_phrases_4b", INC_PHRASES),
        ("nice_inc_list_4b", INC_LIST),
    ]
    ids = {r.review_id for r in round_robin(sample, 120)}
    body = _env_block(
        out_dir, [v for v, _ in variants], "python -m review_llm_eval.experiments run NICE"
    )
    body += [
        "120 reviews (24 per star). 'A phrase' = the review contains an incident phrase "
        "(`cues.py`, deliberately broad).",
        "",
    ]
    rows = []
    for run_id, variant in variants:
        records = [r for r in load_records(out_dir, run_id) if r["review_id"] in ids]
        if not records:
            rows.append([run_id, "pending"])
            continue
        flagged: dict[int, bool] = {}
        for r in records:
            out = analysis.output_of(r)
            if out is None:
                continue
            e = expand(out, variant)
            inc = e["incident"]
            flagged[r["review_id"]] = bool(inc) if isinstance(inc, list) else inc is not None
        low = [rid for rid in flagged if (reviews[rid].rating or 0) <= 2]
        five = [rid for rid in flagged if reviews[rid].rating == 5]
        cue = [rid for rid in flagged if has_cue("incident", reviews[rid].full_text)]
        rows.append(
            [
                run_id, len(flagged),
                sum(flagged[r] for r in low) / len(low) if low else float("nan"),
                sum(flagged[r] for r in five) / len(five) if five else float("nan"),
                sum(flagged[r] for r in cue) / len(cue) if cue else float("nan"),
                sum(flagged[r] and not has_cue("incident", reviews[r].full_text) for r in flagged),
            ]
        )  # fmt: skip
    body += [
        markdown_table(
            [
                "run",
                "valid",
                "incident on 1-2 star",
                "incident on 5 star",
                "incident when a phrase is present",
                "incident without any phrase",
            ],
            rows,
        ),
        "",
    ]
    _write(results, "nice_incident_wording.md", "Incident field: three alternative wordings", body)


def analyze_language(out_dir: Path, results: Path) -> None:
    records = load_records(out_dir, "l11_language_4b")
    body = _env_block(out_dir, ["l11_language_4b"], "python -m review_llm_eval.experiments run L11")
    if not records:
        _write(results, "l11_language.md", "Language: model vs detector", [*body, "Pending.", ""])
        return
    reviews, _ = rows_to_reviews(load_parquet_rows(RAW_PARQUET))
    by_id = {r.review_id: r for r in reviews}
    pairs = []
    no_spanish = no_spanish_es = 0
    for r in records:
        out = analysis.output_of(r)
        if out is None:
            continue
        text = by_id[r["review_id"]].full_text
        pairs.append((detect(text), out["idi"]))
        if language_scores(text)["es"] == 0:
            no_spanish += 1
            no_spanish_es += out["idi"] == "es"
    counts: dict[tuple[str, str], int] = {}
    for p in pairs:
        counts[p] = counts.get(p, 0) + 1
    agree = sum(n for (a, b), n in counts.items() if a == b)
    body += [
        f"All {len(records)} reviews of the dataset that the detector does not classify as "
        "Spanish (many are bilingual). Model = the `idi` field of qwen3:4b.",
        "",
        markdown_table(
            ["detector", "model", "reviews"], [[a, b, n] for (a, b), n in sorted(counts.items())]
        ),
        "",
        f"Model agrees with the detector on {agree} of {len(pairs)}. Of the {no_spanish} "
        f"reviews without a single Spanish function word, the model answered `es` for "
        f"{no_spanish_es}.",
        "",
    ]
    _write(results, "l11_language.md", "Language: model vs detector", body)


def analyze_short_positives(out_dir: Path, results: Path, sample: list[Review]) -> None:
    """Cost of skipping short positive reviews: what the model extracts from them."""
    reviews = {r.review_id: r for r in sample}
    records = load_records(out_dir, "e1_4b")
    body = _env_block(out_dir, ["e1_4b"], "python -m review_llm_eval.experiments analyze all")
    if not records:
        _write(
            results,
            "nice_short_positives.md",
            "Skipping short positive reviews",
            [*body, "Pending.", ""],
        )
        return
    all_reviews, _ = rows_to_reviews(load_parquet_rows(RAW_PARQUET))
    lengths = sorted(len(r.text) for r in all_reviews if (r.rating or 0) >= 4)
    cut = int(percentile(lengths, 25))
    share = sum(n < cut for n in lengths) / len(all_reviews)
    base = hotel_veto({r.hotel for r in all_reviews})
    rows = []
    for label, keep in (("short (< p25 length)", True), ("the other 4-5 star", False)):
        subset_ = [
            r
            for r in records
            if (rv := reviews[r["review_id"]]).rating
            and rv.rating >= 4
            and (len(rv.text) < cut) == keep
        ]
        outs = [(reviews[r["review_id"]], o) for r in subset_ if (o := analysis.output_of(r))]
        expanded = [expand(o, BASE, rv.full_text, veto_for(rv.hotel, base)) for rv, o in outs]
        secs = [
            sum(a["latency_s"] for a in r["extraction"]["attempts"])
            for r in subset_
            if r.get("extraction")
        ]
        rows.append(
            [
                label, len(outs),
                sum(bool(e["staff"]) for e in expanded) / len(outs) if outs else float("nan"),
                sum(e["recommends"] for e in expanded) / len(outs) if outs else float("nan"),
                sum(any(e[a] for a in ALERTS) for e in expanded),
                mean([len(e["opinions"]) for e in expanded]),
                mean(secs),
            ]
        )  # fmt: skip
    body += [
        f"'Short' = 4-5 star reviews under {cut} characters (the 25th percentile of 4-5 star "
        f"lengths); they are {share:.1%} of the {len(all_reviews)} reviews of the dataset. "
        "The dataset has no review under 60 characters, so this is not the same cut as "
        "production.",
        "",
        markdown_table(
            [
                "group",
                "valid",
                "with a staff name",
                "recommends",
                "with an alert",
                "opinions per review",
                "latency s (4 threads)",
            ],
            rows,
        ),
        "",
    ]
    _write(results, "nice_short_positives.md", "Skipping short positive reviews", body)


def analyze_runs_csv(out_dir: Path, results: Path, specs: dict[str, RunSpec]) -> None:
    from review_llm_eval.report import write_csv

    headers = [
        "run_id",
        "experiment",
        "model",
        "digest",
        "variant",
        "subset",
        "workers",
        "n",
        "valid",
        "valid_first",
        "runaway",
        "out_tok_mean",
        "reviews_per_min",
        "ollama",
        "date",
        "other_models_loaded",
        "gpu_before",
        "gpu_after",
    ]
    rows = []
    for run_id, spec in specs.items():
        records = load_records(out_dir, run_id)
        session = _session(out_dir, run_id)
        if not records or not session:
            continue
        s = analysis.run_summary(records, session)
        environment = session.get("environment") or {}
        rows.append(
            [
                run_id, spec.experiment, spec.model, (environment.get("model") or {}).get("digest"),
                spec.variant.name, spec.subset, spec.workers, s["n"], s["valid"],
                s["valid_first_attempt"], s["runaway_reviews"], s["output_tokens_mean"],
                s["reviews_per_min"], environment.get("ollama_version"), session["started_at"][:10],
                ";".join(session.get("foreign_models") or []),
                environment.get("gpu_state_before"), environment.get("gpu_state_after"),
            ]
        )  # fmt: skip
    write_csv(results / "runs.csv", headers, rows)


def analyze_all(out_dir: Path, results: Path, sample: list[Review]) -> None:
    specs = all_specs(out_dir)
    analyze_e1(out_dir, results, sample)
    analyze_quote_support(out_dir, results, sample)
    analyze_pairwise(
        out_dir, results, "e2_short_keys.md", "E2: short or long keys?",
        [("e1_4b", "e2_long_4b", BASE, LONG_KEYS), ("e1_4b", "e10_repeat_4b", BASE, BASE)],
        "python -m review_llm_eval.experiments run E2",
        ["The second row is the repeatability baseline (same prompt twice, E10): a variant",
         "only 'changes the answers' where it falls below that line."],
        sample=sample,
    )  # fmt: skip
    analyze_e3(out_dir, results)
    analyze_e4(out_dir, results)
    analyze_e5(out_dir, results)
    analyze_e6(out_dir, results)
    analyze_e7(out_dir, results)
    analyze_e8(out_dir, results)
    analyze_e9(out_dir, results)
    analyze_pairwise(
        out_dir, results, "e10_repeatability.md", "E10: is temperature 0 repeatable?",
        [
            ("e1_4b", "e10_repeat_4b", BASE, BASE),
            ("e10_serial_a_4b", "e10_serial_b_4b", BASE, BASE),
        ],
        "python -m review_llm_eval.experiments run E10",
        ["Row 1: 200 reviews twice with 4 threads.",
         "Row 2: 100 reviews twice, one request at a time."],
        sample=sample,
    )  # fmt: skip
    if "nice_maxlength_4b" in specs:
        maxlen = specs["nice_maxlength_4b"].variant
        quote_max, summary_max = maxlen.max_lengths or (0, 0)
        analyze_pairwise(
            out_dir, results, "nice_maxlength.md", "maxLength on quotes and summary",
            [("e1_4b", "nice_maxlength_4b", BASE, maxlen), ("e1_4b", "e10_repeat_4b", BASE, BASE)],
            "python -m review_llm_eval.experiments run NICE",
            [f"maxLength (characters): quote {quote_max}, summary {summary_max} = 95th",
             "percentile of the lengths written in e1_4b. Second row: repeatability baseline."],
            sample=sample,
        )  # fmt: skip
    analyze_pairwise(
        out_dir, results, "nice_prompt_order.md", "Instructions first (prefix caching)",
        [("e1_4b", "nice_reorder_4b", BASE, REORDER), ("e1_4b", "e10_repeat_4b", BASE, BASE)],
        "python -m review_llm_eval.experiments run NICE",
        ["`prompt tok` counts the whole prompt even when the server reuses a cached prefix;",
         "the saving shows in `prompt eval s` (mean prompt-processing time of the first",
         "attempt). Second row: repeatability baseline."],
        sample=sample,
    )  # fmt: skip
    analyze_incident_wording(out_dir, results, sample)
    analyze_short_positives(out_dir, results, sample)
    analyze_language(out_dir, results)
    analyze_runs_csv(out_dir, results, specs)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("action", choices=["list", "run", "analyze"])
    parser.add_argument(
        "target", nargs="?", default="all", help="E1..E10, NICE, L11, a run id or all"
    )
    parser.add_argument("--out", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    specs = all_specs(args.out)
    if args.action == "list":
        for run_id, spec in specs.items():
            print(
                f"{spec.experiment:5} {run_id:28} {spec.model:9} {spec.subset:8} "
                f"workers={spec.workers} variant={spec.variant.name}"
            )
        return
    sample = load_sample(args.sample)
    if args.action == "analyze":
        analyze_all(args.out, args.results, sample)
        return
    if args.target == "E4":
        run_e4_induced(sample, args.out)  # qwen3:4b first, then the sources (8b last)
        run_e4(sample, specs, args.out)
        return
    if args.target == "E9":
        run_e9(sample, args.out)
        return
    if args.target in specs:
        chosen = [specs[args.target]]
    elif args.target == "NICE":
        chosen = [*RUNS["NICE"], maxlength_spec(args.out)]
    else:
        chosen = RUNS.get(args.target) or []
    if not chosen:
        raise SystemExit(f"unknown target {args.target}")
    subsets: dict[str, list[Review]] = {}
    for spec in chosen:
        if spec.subset not in subsets:
            subsets[spec.subset] = subset(spec.subset, sample)
        run(spec, subsets[spec.subset], args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
