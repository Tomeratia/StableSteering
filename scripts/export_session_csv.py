"""Export a session's rounds, candidates, and feedback events to tidy CSV files.

Usage:
    python scripts/export_session_csv.py <session_id> [--out-dir output/exports]

Outputs (in --out-dir):
    rounds.csv      — one row per round
    candidates.csv  — one row per candidate (all rounds)
    feedback.csv    — one row per feedback event (includes critique tags)

These files share a common session_id and round_id join key so they can be
loaded into a notebook or spreadsheet without preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.storage.repository import JsonRepository


def export_session(session_id: str, out_dir: Path) -> None:
    repo = JsonRepository()
    session = repo.get_session(session_id)
    if session is None:
        print(f"Session not found: {session_id}")
        sys.exit(1)

    rounds = repo.list_rounds_for_session(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── rounds.csv ────────────────────────────────────────────────────────────
    rounds_path = out_dir / "rounds.csv"
    with rounds_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "round_id", "round_index",
            "render_status", "latency_ms",
            "winner_candidate_id", "updater", "method",
            "positive_tag_count", "negative_tag_count",
            "total_positive_weight", "total_negative_weight",
            "converged", "created_at",
        ])
        for rnd in rounds:
            summary = rnd.update_summary or {}
            feedback = rnd.feedback_events[0] if rnd.feedback_events else None
            writer.writerow([
                session_id,
                rnd.id,
                rnd.round_index,
                rnd.render_status.value,
                rnd.latency_ms,
                summary.get("winner_candidate_id", ""),
                summary.get("updater", ""),
                summary.get("method", ""),
                summary.get("positive_tag_count", summary.get("total_positive_weight", "")),
                summary.get("negative_tag_count", summary.get("total_negative_weight", "")),
                summary.get("total_positive_weight", ""),
                summary.get("total_negative_weight", ""),
                session.converged,
                rnd.created_at.isoformat(),
            ])
    print(f"  rounds.csv     → {rounds_path} ({len(rounds)} rows)")

    # ── candidates.csv ────────────────────────────────────────────────────────
    candidates_path = out_dir / "candidates.csv"
    total_candidates = 0
    with candidates_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "round_id", "round_index",
            "candidate_id", "candidate_index", "sampler_role",
            "seed", "z",
        ])
        for rnd in rounds:
            for cand in rnd.candidates:
                writer.writerow([
                    session_id,
                    rnd.id,
                    rnd.round_index,
                    cand.id,
                    cand.candidate_index,
                    cand.sampler_role,
                    cand.seed,
                    json.dumps(cand.z),
                ])
                total_candidates += 1
    print(f"  candidates.csv → {candidates_path} ({total_candidates} rows)")

    # ── feedback.csv ──────────────────────────────────────────────────────────
    feedback_path = out_dir / "feedback.csv"
    total_events = 0
    with feedback_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_id", "round_id", "round_index",
            "feedback_type", "winner_candidate_id",
            "ratings_json", "critique_tags_json",
        ])
        for rnd in rounds:
            for event in rnd.feedback_events:
                payload = event.normalized_payload or {}
                writer.writerow([
                    session_id,
                    rnd.id,
                    rnd.round_index,
                    event.type.value,
                    payload.get("winner_candidate_id", ""),
                    json.dumps(payload.get("ratings", {})),
                    json.dumps(payload.get("critique_tags", {})),
                ])
                total_events += 1
    print(f"  feedback.csv   → {feedback_path} ({total_events} rows)")

    print(f"\nExport complete: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a session to CSV files.")
    parser.add_argument("session_id", help="Session ID (e.g. ses_abc123)")
    parser.add_argument("--out-dir", default="output/exports", help="Output directory")
    args = parser.parse_args()

    export_session(args.session_id, Path(args.out_dir) / args.session_id)


if __name__ == "__main__":
    main()
