from __future__ import annotations

from app.core.schema import Candidate, FeedbackEvent, Round, Session
from app.samplers.base import clamp_vector
from app.updaters.critique_weighted_pref import NEGATIVE_TAGS, POSITIVE_TAGS

# Decay factor per round of age. Tags selected 1 round ago count fully;
# tags selected 2 rounds ago count MOMENTUM_DECAY of that; and so on.
MOMENTUM_DECAY = 0.75


def _build_tag_frequencies(past_rounds: list[Round], current_round_index: int) -> dict[str, float]:
    """Return a decay-weighted frequency map over past rounds.

    Tags in round r contribute MOMENTUM_DECAY^(current - r) to their score,
    so recent rounds carry more influence than old ones.
    """
    freq: dict[str, float] = {}
    for rnd in past_rounds:
        age = current_round_index - rnd.round_index
        if age <= 0:
            continue
        decay = MOMENTUM_DECAY ** age
        for event in rnd.feedback_events:
            critique_tags: dict[str, list[str]] = event.normalized_payload.get("critique_tags", {})
            for tags in critique_tags.values():
                for tag in tags:
                    freq[tag] = freq.get(tag, 0.0) + decay
    return freq


class CritiqueMomentumPreferenceUpdater:
    """Updater that weights critique tags by their historical frequency.

    Replaces the flat +1/-1 tag weight of CritiqueWeightedPreferenceUpdater
    with a decay-weighted frequency score accumulated across all previous rounds.
    A tag the user selects consistently carries more influence than one selected
    only once, making the steering vector increasingly sensitive to repeated signals.

    Math:
        freq(t) = sum_r MOMENTUM_DECAY^(current - r) * I[t in tags_r]
        weight_i = sum(freq(t) for positive tags) - sum(freq(t) for negative tags)

    Falls back to the rated winner when no tags are present.
    """

    name = "critique_momentum_preference"

    def update(
        self,
        session: Session,
        candidates: list[Candidate],
        feedback: FeedbackEvent,
        past_rounds: list[Round] | None = None,
    ) -> tuple[list[float], dict]:
        dimensions = len(session.current_z)
        candidate_map = {c.id: c for c in candidates}
        critique_tags: dict[str, list[str]] = feedback.normalized_payload.get("critique_tags", {})

        tag_freq = _build_tag_frequencies(past_rounds or [], session.current_round)

        positive_terms: list[tuple[float, Candidate]] = []
        negative_terms: list[tuple[float, Candidate]] = []
        total_positive_weight = 0.0
        total_negative_weight = 0.0

        for candidate_id, tags in critique_tags.items():
            candidate = candidate_map.get(candidate_id)
            if candidate is None:
                continue
            pos_weight = sum(tag_freq.get(t, 1.0) for t in tags if t in POSITIVE_TAGS)
            neg_weight = sum(tag_freq.get(t, 1.0) for t in tags if t in NEGATIVE_TAGS)
            net = pos_weight - neg_weight
            total_positive_weight += pos_weight
            total_negative_weight += neg_weight
            if net > 0:
                positive_terms.append((net, candidate))
            elif net < 0:
                negative_terms.append((-net, candidate))

        if not positive_terms:
            winner_id = feedback.normalized_payload["winner_candidate_id"]
            winner = candidate_map.get(winner_id)
            if winner is not None:
                positive_terms.append((1.0, winner))

        positive_center = self._weighted_centroid(positive_terms, dimensions)
        negative_center = self._weighted_centroid(negative_terms, dimensions)
        direction = [pos - neg for pos, neg in zip(positive_center, negative_center, strict=False)]

        alpha = 0.55 if negative_terms and total_positive_weight else 0.4
        updated = clamp_vector(
            [round(z + alpha * d, 6) for z, d in zip(session.current_z, direction, strict=False)],
            session.config.trust_radius,
        )
        return updated, {
            "updater": self.name,
            "winner_candidate_id": feedback.normalized_payload["winner_candidate_id"],
            "method": "critique_momentum_move",
            "total_positive_weight": round(total_positive_weight, 4),
            "total_negative_weight": round(total_negative_weight, 4),
            "momentum_decay": MOMENTUM_DECAY,
            "rounds_with_history": len(past_rounds) if past_rounds else 0,
        }

    @staticmethod
    def _weighted_centroid(terms: list[tuple[float, Candidate]], dimensions: int) -> list[float]:
        if not terms:
            return [0.0] * dimensions
        total_weight = sum(w for w, _ in terms)
        if total_weight <= 0:
            return [0.0] * dimensions
        values = [0.0] * dimensions
        for weight, candidate in terms:
            for i, v in enumerate(candidate.z):
                if i < dimensions:
                    values[i] += weight * v
        return [v / total_weight for v in values]
