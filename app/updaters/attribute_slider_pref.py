from __future__ import annotations

from app.core.schema import Candidate, FeedbackEvent, Session
from app.samplers.base import clamp_vector

# Default attribute labels used when the session prompt doesn't supply custom ones.
# These map to steering dimensions by index (dim 0 = brightness, etc.).
DEFAULT_ATTRIBUTES = ["brightness", "style", "detail", "composition", "mood"]


class AttributeSliderUpdater:
    """Updater that moves z directly along user-specified attribute directions.

    Unlike rating-based updaters that infer direction from candidate comparisons,
    this updater lets the user state explicitly where they want to go via sliders.
    Each slider value in [-1, 1] is treated as a signed displacement along the
    corresponding steering dimension.

    Math:
        direction[i] = sliders.get(attribute_i, 0.0)
        z_new = clamp(z_old + alpha * direction, trust_radius)

    This is the most direct possible steering signal — the user controls z
    without the system having to infer anything from preference comparisons.
    """

    name = "attribute_slider_preference"

    def update(self, session: Session, candidates: list[Candidate], feedback: FeedbackEvent) -> tuple[list[float], dict]:
        dimensions = len(session.current_z)
        sliders: dict[str, float] = feedback.normalized_payload.get("sliders", {})

        # Map attribute names to dimension indices using DEFAULT_ATTRIBUTES order.
        direction = [0.0] * dimensions
        active_attrs: list[str] = []
        for dim_index, attr_name in enumerate(DEFAULT_ATTRIBUTES):
            if dim_index >= dimensions:
                break
            if attr_name in sliders:
                direction[dim_index] = sliders[attr_name]
                active_attrs.append(attr_name)

        alpha = 0.5
        updated = clamp_vector(
            [round(z + alpha * d, 6) for z, d in zip(session.current_z, direction, strict=False)],
            session.config.trust_radius,
        )

        winner_id = feedback.normalized_payload.get("winner_candidate_id", "")
        return updated, {
            "updater": self.name,
            "winner_candidate_id": winner_id,
            "method": "attribute_slider_move",
            "active_attributes": active_attrs,
            "slider_values": sliders,
        }
