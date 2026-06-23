from __future__ import annotations

from app.core.schema import Candidate, FeedbackEvent, Round, Session, StrategyConfig
from app.updaters.critique_momentum_pref import CritiqueMomentumPreferenceUpdater, _build_tag_frequencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(current_z: list[float]) -> Session:
    config = StrategyConfig(updater="critique_momentum_preference", trust_radius=0.55)
    return Session(
        experiment_id="exp_unit",
        prompt="unit prompt",
        model_name=config.model_name,
        config=config,
        current_z=current_z,
        current_round=3,
    )


def _candidate(z: list[float]) -> Candidate:
    return Candidate(
        round_id="rnd_unit",
        candidate_index=0,
        z=z,
        sampler_role="incumbent",
        seed=1,
        generation_params={},
    )


def _feedback_event(winner_id: str, tags: dict) -> FeedbackEvent:
    return FeedbackEvent(
        round_id="rnd_unit",
        type="critique_rating",
        payload={},
        normalized_payload={"winner_candidate_id": winner_id, "ratings": {}, "critique_tags": tags},
        critique_tags=tags,
    )


def _round(round_index: int, tags: dict, winner_id: str = "c1") -> Round:
    rnd = Round(
        session_id="ses_unit",
        round_index=round_index,
        incumbent_z=[0.0, 0.0],
        trust_radius=0.55,
        seed_policy="fixed-per-candidate",
    )
    rnd.feedback_events = [_feedback_event(winner_id, tags)]
    return rnd


# ---------------------------------------------------------------------------
# Unit tests for _build_tag_frequencies
# ---------------------------------------------------------------------------


def test_tag_frequency_accumulates_across_rounds() -> None:
    rounds = [
        _round(1, {"c1": ["too_dark"]}),
        _round(2, {"c1": ["too_dark", "good_color"]}),
    ]
    freq = _build_tag_frequencies(rounds, current_round_index=3)

    assert "too_dark" in freq
    assert "good_color" in freq
    assert freq["too_dark"] > freq["good_color"]


def test_tag_frequency_decays_older_rounds() -> None:
    rounds = [
        _round(1, {"c1": ["too_dark"]}),
        _round(3, {"c1": ["too_dark"]}),
    ]
    freq = _build_tag_frequencies(rounds, current_round_index=4)
    # Round 3 (age=1) contributes 0.75; round 1 (age=3) contributes 0.75^3 ≈ 0.42
    assert freq["too_dark"] > 0.75


def test_tag_frequency_skips_current_round() -> None:
    rounds = [_round(2, {"c1": ["good_color"]})]
    freq = _build_tag_frequencies(rounds, current_round_index=2)
    assert "good_color" not in freq


# ---------------------------------------------------------------------------
# Unit tests for CritiqueMomentumPreferenceUpdater
# ---------------------------------------------------------------------------


def test_momentum_updater_moves_toward_positive() -> None:
    pos = _candidate([0.5, 0.0])
    neg = _candidate([-0.5, 0.0])
    event = _feedback_event(pos.id, {pos.id: ["good_color"], neg.id: ["too_dark"]})
    updated, summary = CritiqueMomentumPreferenceUpdater().update(_session([0.0, 0.0]), [pos, neg], event)

    assert updated[0] > 0.0
    assert summary["method"] == "critique_momentum_move"


def test_momentum_updater_amplifies_with_history() -> None:
    """After consistent tags, momentum should produce a larger or equal step."""
    pos = _candidate([0.5, 0.0])
    neg = _candidate([-0.5, 0.0])

    past_rounds = [
        _round(1, {pos.id: ["good_color"], neg.id: ["too_dark"]}, winner_id=pos.id),
        _round(2, {pos.id: ["good_color"], neg.id: ["too_dark"]}, winner_id=pos.id),
    ]
    event = _feedback_event(pos.id, {pos.id: ["good_color"], neg.id: ["too_dark"]})

    z_with_history, _ = CritiqueMomentumPreferenceUpdater().update(
        _session([0.0, 0.0]), [pos, neg], event, past_rounds=past_rounds
    )
    z_no_history, _ = CritiqueMomentumPreferenceUpdater().update(
        _session([0.0, 0.0]), [pos, neg], event, past_rounds=[]
    )

    assert abs(z_with_history[0]) >= abs(z_no_history[0])


def test_momentum_updater_falls_back_to_winner_without_tags() -> None:
    winner = _candidate([0.4, 0.0])
    other = _candidate([-0.4, 0.0])
    event = _feedback_event(winner.id, {})
    updated, _ = CritiqueMomentumPreferenceUpdater().update(_session([0.0, 0.0]), [winner, other], event)

    assert updated[0] > 0.0


def test_momentum_weights_are_larger_than_flat_with_history() -> None:
    """With history, momentum tag weights must exceed the flat weight of 1.0 per tag."""
    pos = _candidate([0.5, 0.0])
    neg = _candidate([-0.5, 0.0])

    past_rounds = [
        _round(1, {pos.id: ["good_color"], neg.id: ["too_dark"]}, winner_id=pos.id),
        _round(2, {pos.id: ["good_color"], neg.id: ["too_dark"]}, winner_id=pos.id),
    ]
    event = _feedback_event(pos.id, {pos.id: ["good_color"], neg.id: ["too_dark"]})

    _, summary_momentum = CritiqueMomentumPreferenceUpdater().update(
        _session([0.0, 0.0]), [pos, neg], event, past_rounds=past_rounds
    )
    _, summary_no_history = CritiqueMomentumPreferenceUpdater().update(
        _session([0.0, 0.0]), [pos, neg], event, past_rounds=[]
    )

    # With history the accumulated frequency exceeds 1.0 (the flat default),
    # so total weights must be strictly larger.
    assert summary_momentum["total_positive_weight"] > summary_no_history["total_positive_weight"]
    assert summary_momentum["total_negative_weight"] > summary_no_history["total_negative_weight"]
