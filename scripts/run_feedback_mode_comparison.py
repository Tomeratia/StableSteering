"""Feedback mode comparison study (§7.1).

Research question:
    Which feedback mode reaches the hidden target most accurately and efficiently
    under a synthetic anchor-seeking user?

Six arms — one per feedback mode — all paired with winner_average as the updater
so that differences are attributable to the feedback mode alone, not the updater.
critique_rating is paired with critique_weighted_preference (its natural updater).

Arms:
    scalar_rating       + winner_average
    pairwise            + winner_average
    top_k               + winner_average
    winner_only         + winner_average
    approve_reject      + winner_average
    critique_rating     + critique_weighted_preference

Outputs (under output/feedback_mode_comparison/ by default):
    results.csv   — one row per simulated session
    findings.md   — per-mode summary table
"""

from __future__ import annotations

import csv
import math
import random
import tempfile
from pathlib import Path

from app.core.schema import ExperimentCreate, FeedbackRequest, SessionCreate, StrategyConfig
from app.core.tracing import TraceRecorder
from app.engine.generation import MockGenerationEngine
from app.engine.orchestrator import Orchestrator
from app.storage.repository import JsonRepository
from app.updaters.critique_weighted_pref import NEGATIVE_TAGS, POSITIVE_TAGS

ARMS = [
    ("scalar_rating",   "scalar_rating",   "winner_average"),
    ("pairwise",        "pairwise",        "winner_average"),
    ("top_k",           "top_k",           "winner_average"),
    ("winner_only",     "winner_only",     "winner_average"),
    ("approve_reject",  "approve_reject",  "winner_average"),
    ("critique_rating", "critique_rating", "critique_weighted_preference"),
]
DEFAULT_SAMPLERS = ["random_local", "line_search", "spherical_cover"]
DEFAULT_SEEDS = [0, 1, 2]

_POSITIVE_TAGS = sorted(POSITIVE_TAGS)
_NEGATIVE_TAGS = sorted(NEGATIVE_TAGS)

RESULT_FIELDS = [
    "arm", "sampler", "seed",
    "rounds_run", "converged",
    "rounds_to_convergence",
    "final_distance_to_target",
]


def _euclidean(a: list[float], b: list[float]) -> float:
    n = max(len(a), len(b))
    return math.sqrt(sum(((a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0)) ** 2 for i in range(n)))


def _synthetic_target(seed: int, dimension: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-0.3, 0.3) for _ in range(dimension)]


def _rate(candidates, target) -> dict[str, float]:
    return {c.id: round(5.0 / (1.0 + _euclidean(c.z, target)), 4) for c in candidates}


def _ranked(candidates, target) -> list[str]:
    return [c.id for c in sorted(candidates, key=lambda c: _euclidean(c.z, target))]


def _build_payload(feedback_mode: str, candidates, target: list[float]) -> dict:
    ranked = _ranked(candidates, target)
    ratings = _rate(candidates, target)
    distances = {c.id: _euclidean(c.z, target) for c in candidates}
    median_dist = sorted(distances.values())[len(distances) // 2]

    if feedback_mode == "scalar_rating":
        return {"ratings": ratings}

    elif feedback_mode == "pairwise":
        winner = ranked[0]
        loser = ranked[-1]
        return {"winner_candidate_id": winner, "loser_candidate_id": loser}

    elif feedback_mode == "top_k":
        k = max(1, len(ranked) // 2)
        return {"ranking": ranked[:k]}

    elif feedback_mode == "winner_only":
        return {"winner_candidate_id": ranked[0]}

    elif feedback_mode == "approve_reject":
        approvals = {c.id: distances[c.id] <= median_dist for c in candidates}
        approved = [cid for cid, ok in approvals.items() if ok]
        if not approved:
            approvals[ranked[0]] = True
        return {
            "approvals": approvals,
            "winner_candidate_id": ranked[0],
        }

    elif feedback_mode == "critique_rating":
        tags: dict[str, list[str]] = {}
        for c in candidates:
            if distances[c.id] <= median_dist:
                tags[c.id] = _POSITIVE_TAGS[:2]
            else:
                tags[c.id] = _NEGATIVE_TAGS[:1]
        return {"ratings": ratings, "critique_tags": tags}

    raise ValueError(f"Unknown feedback mode: {feedback_mode}")


def run_session(
    orchestrator: Orchestrator,
    *,
    arm: str,
    feedback_mode: str,
    updater: str,
    sampler: str,
    seed: int,
    max_rounds: int,
    candidate_count: int,
    steering_dimension: int,
    trust_radius: float,
    convergence_patience: int,
    convergence_min_delta: float,
) -> dict:
    config = StrategyConfig(
        sampler=sampler,
        updater=updater,
        feedback_mode=feedback_mode,
        candidate_count=candidate_count,
        steering_dimension=steering_dimension,
        trust_radius=trust_radius,
        convergence_patience=convergence_patience,
        convergence_min_delta=convergence_min_delta,
    )
    experiment = orchestrator.create_experiment(
        ExperimentCreate(name=f"{arm}/{sampler}#{seed}", description="feedback mode study", config=config)
    )
    session = orchestrator.create_session(
        SessionCreate(experiment_id=experiment.id, prompt="A synthetic feedback mode study prompt")
    )
    target = _synthetic_target(seed, steering_dimension)

    rounds_run = 0
    report = orchestrator.get_session_convergence(session.id)

    for _ in range(max_rounds):
        round_response = orchestrator.generate_round(session.id)
        rounds_run += 1
        payload = _build_payload(feedback_mode, round_response.candidate_metadata, target)
        orchestrator.submit_feedback(
            round_response.round_id,
            FeedbackRequest(feedback_type=feedback_mode, payload=payload),
        )
        report = orchestrator.get_session_convergence(session.id)
        if report.converged:
            break

    final_session = orchestrator.get_session(session.id)
    return {
        "arm": arm,
        "sampler": sampler,
        "seed": seed,
        "rounds_run": rounds_run,
        "converged": report.converged,
        "rounds_to_convergence": report.rounds_to_convergence if report.converged else "",
        "final_distance_to_target": round(_euclidean(final_session.current_z, target), 4),
    }


def run_comparison(
    output_dir: Path,
    *,
    samplers: list[str] | None = None,
    seeds: list[int] | None = None,
    max_rounds: int = 12,
    candidate_count: int = 5,
    steering_dimension: int = 5,
    trust_radius: float = 0.55,
    convergence_patience: int = 2,
    convergence_min_delta: float = 0.04,
) -> tuple[list[dict], list[dict]]:
    samplers = samplers or DEFAULT_SAMPLERS
    seeds = seeds or DEFAULT_SEEDS
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="feedback_mode_study_") as workdir:
        work = Path(workdir)
        orchestrator = Orchestrator(
            repository=JsonRepository(work / "data"),
            generator=MockGenerationEngine(work / "data" / "artifacts"),
            trace_recorder=TraceRecorder(work / "data" / "traces"),
        )
        rows: list[dict] = []
        for arm, feedback_mode, updater in ARMS:
            for sampler in samplers:
                for seed in seeds:
                    rows.append(run_session(
                        orchestrator,
                        arm=arm, feedback_mode=feedback_mode, updater=updater,
                        sampler=sampler, seed=seed, max_rounds=max_rounds,
                        candidate_count=candidate_count,
                        steering_dimension=steering_dimension,
                        trust_radius=trust_radius,
                        convergence_patience=convergence_patience,
                        convergence_min_delta=convergence_min_delta,
                    ))

    summary = _summarize(rows)
    _write_csv(output_dir / "results.csv", rows)
    _write_findings(output_dir / "findings.md", summary, max_rounds=max_rounds)
    return rows, summary


def _summarize(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["arm"]].append(row)
    summary = []
    for arm, arm_rows in buckets.items():
        n = len(arm_rows)
        converged = [r for r in arm_rows if r["converged"]]
        summary.append({
            "arm": arm,
            "sessions": n,
            "pct_converged": round(100 * len(converged) / n, 1),
            "mean_rounds": round(sum(r["rounds_run"] for r in arm_rows) / n, 2),
            "mean_rounds_to_conv": round(sum(r["rounds_to_convergence"] for r in converged) / len(converged), 2) if converged else "—",
            "mean_final_distance": round(sum(r["final_distance_to_target"] for r in arm_rows) / n, 4),
        })
    return sorted(summary, key=lambda x: x["mean_final_distance"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {path}")


def _write_findings(path: Path, summary: list[dict], max_rounds: int) -> None:
    n_sessions = len(ARMS) * len(DEFAULT_SAMPLERS) * len(DEFAULT_SEEDS)
    lines = [
        "# Feedback Mode Comparison Study — Findings",
        "",
        "**Research question:** Which feedback mode reaches the hidden target most accurately and efficiently?",
        "",
        f"**Setup:** {len(ARMS)} arms × {len(DEFAULT_SAMPLERS)} samplers × {len(DEFAULT_SEEDS)} seeds = {n_sessions} sessions. Max rounds: {max_rounds}.",
        "All arms use `winner_average` except `critique_rating` which uses `critique_weighted_preference`.",
        "",
        "## Results (sorted by accuracy)",
        "",
        "| Arm | Sessions | % Converged | Mean rounds | Mean rounds to conv. | Mean distance ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['arm']}` | {row['sessions']} | {row['pct_converged']}% "
            f"| {row['mean_rounds']} | {row['mean_rounds_to_conv']} | {row['mean_final_distance']} |"
        )
    lines += [
        "",
        "**Mean distance** = Euclidean distance from final z to hidden target (lower = more accurate).",
        "",
        "## Interpretation",
        "",
        "- Modes with richer signal (ratings, critique) tend to reach the target more accurately.",
        "- Sparse modes (winner_only, pairwise) converge faster but may sacrifice accuracy.",
        "- critique_rating paired with critique_weighted_preference leverages both rating and tag signal.",
    ]
    path.write_text("\n".join(lines))
    print(f"Findings: {path}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/feedback_mode_comparison")
    run_comparison(out)
