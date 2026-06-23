from __future__ import annotations

from app.core.schema import Candidate, FeedbackEvent, Session
from app.samplers.base import clamp_vector


class BestVsIncumbentUpdater:
    """Updater for the best-vs-incumbent feedback mode.

    The user is shown exactly two candidates:
      - the incumbent (previous round winner, carried forward)
      - the best challenger from this round (chosen by the sampler)

    If the user picks the challenger  → z moves toward the challenger.
    If the user keeps the incumbent   → z stays at incumbent (no change).

    This is a stricter update rule than winner_average: the incumbent is
    the explicit baseline, and only a clear preference for the challenger
    causes z to move. Ties and uncertain rounds produce no movement.

    Math:
        if challenger wins:
            direction = z_challenger - z_incumbent
            z_new = clamp(z_old + alpha * direction, trust_radius)
        else:
            z_new = z_old   (incumbent retained, no update)
    """

    name = "best_vs_incumbent_preference"

    def update(self, session: Session, candidates: list[Candidate], feedback: FeedbackEvent) -> tuple[list[float], dict]:
        payload = feedback.normalized_payload
        winner_id = payload.get("winner_candidate_id", "")
        incumbent_id = payload.get("incumbent_candidate_id", "")
        kept_incumbent = payload.get("kept_incumbent", winner_id == incumbent_id)

        candidate_map = {c.id: c for c in candidates}
        winner = candidate_map.get(winner_id)
        incumbent = candidate_map.get(incumbent_id)

        # No incumbent yet (first round) — move toward winner directly.
        if incumbent is None or not incumbent_id:
            alpha = 0.5
            direction = [w - z for w, z in zip(winner.z, session.current_z, strict=False)]
            updated = clamp_vector(
                [round(z + alpha * d, 6) for z, d in zip(session.current_z, direction, strict=False)],
                session.config.trust_radius,
            )
            return updated, {
                "updater": self.name,
                "winner_candidate_id": winner_id,
                "method": "challenger_accepted",
                "kept_incumbent": False,
            }

        if kept_incumbent or winner is None or winner_id == incumbent_id:
            return session.current_z, {
                "updater": self.name,
                "winner_candidate_id": winner_id,
                "method": "incumbent_retained",
                "kept_incumbent": True,
            }

        alpha = 0.5
        direction = [w - i for w, i in zip(winner.z, incumbent.z, strict=False)]
        updated = clamp_vector(
            [round(z + alpha * d, 6) for z, d in zip(session.current_z, direction, strict=False)],
            session.config.trust_radius,
        )

        return updated, {
            "updater": self.name,
            "winner_candidate_id": winner_id,
            "method": "challenger_accepted",
            "kept_incumbent": False,
            "incumbent_candidate_id": incumbent_id,
        }
