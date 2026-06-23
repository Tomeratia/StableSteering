"""Diversity-seeking synthetic user study (§8.2).

Unlike the anchor-seeking study (run_critique_comparison.py), a diversity-seeking
synthetic user does NOT have a single hidden target. Instead it rewards candidates
that are far apart from each other — it wants to explore the steering space, not
converge to one point.

Research question:
    Does the critique_momentum_preference updater explore the steering space more
    broadly (higher inter-candidate diversity) than the flat critique_weighted or
    plain winner_average updaters under a diversity-seeking user?

Three arms:
    baseline_scalar   — scalar_rating + winner_average
    critique_weighted — critique_rating + critique_weighted_preference
    critique_momentum — critique_rating + critique_momentum_preference

Outputs (under output/diversity_comparison/ by default):
    results.csv   — one row per simulated session
    findings.md   — per-strategy summary with diversity metrics
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
    ("baseline_scalar",   "scalar_rating",   "winner_average"),
    ("critique_weighted", "critique_rating",  "critique_weighted_preference"),
    ("critique_momentum", "critique_rating",  "critique_momentum_preference"),
]
DEFAULT_SAMPLERS = ["random_local", "line_search", "spherical_cover"]
DEFAULT_SEEDS = [0, 1, 2]

_POSITIVE_TAGS = sorted(POSITIVE_TAGS)
_NEGATIVE_TAGS = sorted(NEGATIVE_TAGS)

RESULT_FIELDS = [
    "arm", "sampler", "seed",
    "rounds_run", "converged",
    "mean_inter_candidate_distance",
    "final_z_magnitude",
]


def _distance(a: list[float], b: list[float]) -> float:
    n = max(len(a), len(b))
    return math.sqrt(sum((a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0) for i in range(n)) ** 2
                     for _ in [None])  # type: ignore[misc]


def _euclidean(a: list[float], b: list[float]) -> float:
    n = max(len(a), len(b))
    return math.sqrt(sum(((a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0)) ** 2 for i in range(n)))


def _inter_candidate_diversity(candidates) -> float:
    """Mean pairwise Euclidean distance between all candidates in a round."""
    zs = [c.z for c in candidates]
    if len(zs) < 2:
        return 0.0
    pairs = [(zs[i], zs[j]) for i in range(len(zs)) for j in range(i + 1, len(zs))]
    return sum(_euclidean(a, b) for a, b in pairs) / len(pairs)


def _diversity_ratings(candidates) -> dict[str, float]:
    """Rate candidates higher when they are farther from all others (diversity reward)."""
    zs = {c.id: c.z for c in candidates}
    scores: dict[str, float] = {}
    for c in candidates:
        mean_dist = sum(_euclidean(c.z, zs[other.id]) for other in candidates if other.id != c.id) / max(len(candidates) - 1, 1)
        # Map to 1-5 range: farther = higher rating
        scores[c.id] = round(1.0 + 4.0 * min(mean_dist / 0.5, 1.0), 4)
    return scores


def _diversity_critique_tags(candidates) -> dict[str, list[str]]:
    """Emit tags aligned with diversity: isolated candidates get positive tags."""
    zs = {c.id: c.z for c in candidates}
    mean_dists = {
        c.id: sum(_euclidean(c.z, zs[o.id]) for o in candidates if o.id != c.id) / max(len(candidates) - 1, 1)
        for c in candidates
    }
    median = sorted(mean_dists.values())[len(mean_dists) // 2]
    tags: dict[str, list[str]] = {}
    for c in candidates:
        if mean_dists[c.id] >= median:
            tags[c.id] = _POSITIVE_TAGS[:2]
        else:
            tags[c.id] = _NEGATIVE_TAGS[:1]
    return tags


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
        ExperimentCreate(name=f"{arm}/{sampler}#{seed}", description="diversity study", config=config)
    )
    session = orchestrator.create_session(
        SessionCreate(experiment_id=experiment.id, prompt="A synthetic diversity study prompt")
    )

    rounds_run = 0
    diversity_scores: list[float] = []
    report = orchestrator.get_session_convergence(session.id)

    for _ in range(max_rounds):
        round_response = orchestrator.generate_round(session.id)
        rounds_run += 1
        candidates = round_response.candidate_metadata
        diversity_scores.append(_inter_candidate_diversity(candidates))

        ratings = _diversity_ratings(candidates)
        if feedback_mode == "critique_rating":
            payload = {"ratings": ratings, "critique_tags": _diversity_critique_tags(candidates)}
        else:
            payload = {"ratings": ratings}

        orchestrator.submit_feedback(
            round_response.round_id,
            FeedbackRequest(feedback_type=feedback_mode, payload=payload),
        )
        report = orchestrator.get_session_convergence(session.id)
        if report.converged:
            break

    final_session = orchestrator.get_session(session.id)
    magnitude = math.sqrt(sum(v ** 2 for v in final_session.current_z))

    return {
        "arm": arm,
        "sampler": sampler,
        "seed": seed,
        "rounds_run": rounds_run,
        "converged": report.converged,
        "mean_inter_candidate_distance": round(sum(diversity_scores) / len(diversity_scores), 4) if diversity_scores else 0.0,
        "final_z_magnitude": round(magnitude, 4),
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

    with tempfile.TemporaryDirectory(prefix="diversity_study_") as workdir:
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
        summary.append({
            "arm": arm,
            "sessions": n,
            "mean_rounds": round(sum(r["rounds_run"] for r in arm_rows) / n, 2),
            "pct_converged": round(100 * sum(r["converged"] for r in arm_rows) / n, 1),
            "mean_diversity": round(sum(r["mean_inter_candidate_distance"] for r in arm_rows) / n, 4),
            "mean_z_magnitude": round(sum(r["final_z_magnitude"] for r in arm_rows) / n, 4),
        })
    return sorted(summary, key=lambda x: -x["mean_diversity"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {path}")


def _write_findings(path: Path, summary: list[dict], max_rounds: int) -> None:
    lines = [
        "# Diversity Comparison Study — Findings",
        "",
        "**Research question:** Does critique-assisted feedback produce more diverse candidate exploration than plain scalar ratings under a diversity-seeking synthetic user?",
        "",
        f"**Setup:** {len(ARMS)} arms × {len(DEFAULT_SAMPLERS)} samplers × {len(DEFAULT_SEEDS)} seeds = {len(ARMS) * len(DEFAULT_SAMPLERS) * len(DEFAULT_SEEDS)} sessions. Max rounds: {max_rounds}.",
        "",
        "## Results",
        "",
        "| Arm | Sessions | % Converged | Mean rounds | Mean diversity | Mean ‖z‖ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['arm']}` | {row['sessions']} | {row['pct_converged']}% "
            f"| {row['mean_rounds']} | {row['mean_diversity']} | {row['mean_z_magnitude']} |"
        )
    lines += [
        "",
        "**Mean diversity** = average inter-candidate Euclidean distance per round (higher = more exploratory).",
        "**Mean ‖z‖** = magnitude of final steering vector (higher = moved farther from origin).",
        "",
        "## Interpretation",
        "",
        "A higher mean diversity score indicates that the feedback mode steers the system",
        "toward broader exploration rather than premature convergence to a single region.",
        "The momentum updater accumulates tag history, which may reinforce diverse directions",
        "over time compared to flat weighting.",
    ]
    path.write_text("\n".join(lines))
    print(f"Findings: {path}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/diversity_comparison")
    run_comparison(out)
