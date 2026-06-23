"""Tests for best_vs_incumbent feedback mode and updater."""
from __future__ import annotations

import pytest

from app.core.schema import FeedbackEvent, FeedbackType, Session, StrategyConfig
from app.feedback.normalization import normalize_feedback, _normalize_best_vs_incumbent
from app.updaters.best_vs_incumbent_pref import BestVsIncumbentUpdater
from app.core.schema import Candidate
from app.updaters.winner_average import WinnerAverageUpdater


def make_candidate(cid, z, sampler_role="explorer"):
    return Candidate(id=cid, round_id="r1", candidate_index=0, z=z, sampler_role=sampler_role,
                     seed=0, generation_params={})


def make_session(z=None):
    cfg = StrategyConfig(trust_radius=0.55, steering_dimension=3)
    z = z or [0.0, 0.0, 0.0]
    return Session(experiment_id="e1", prompt="test", config=cfg, current_z=z, model_name="mock")


def make_feedback_event(winner_id, incumbent_id, challenger_id, kept_incumbent):
    return FeedbackEvent(
        round_id="r1",
        type=FeedbackType.best_vs_incumbent,
        payload={},
        normalized_payload={
            "winner_candidate_id": winner_id,
            "incumbent_candidate_id": incumbent_id,
            "challenger_candidate_id": challenger_id,
            "kept_incumbent": kept_incumbent,
        },
    )


# --- normalization ---

def test_normalize_requires_winner():
    with pytest.raises(ValueError, match="winner_candidate_id"):
        _normalize_best_vs_incumbent({})


def test_normalize_sets_kept_incumbent_true_when_winner_is_incumbent():
    result = _normalize_best_vs_incumbent({
        "winner_candidate_id": "inc",
        "incumbent_candidate_id": "inc",
        "challenger_candidate_id": "chal",
    })
    assert result["kept_incumbent"] is True


def test_normalize_sets_kept_incumbent_false_when_winner_is_challenger():
    result = _normalize_best_vs_incumbent({
        "winner_candidate_id": "chal",
        "incumbent_candidate_id": "inc",
        "challenger_candidate_id": "chal",
    })
    assert result["kept_incumbent"] is False


def test_normalize_via_normalize_feedback():
    from app.core.schema import FeedbackRequest
    req = FeedbackRequest(
        feedback_type="best_vs_incumbent",
        payload={
            "winner_candidate_id": "chal",
            "incumbent_candidate_id": "inc",
            "challenger_candidate_id": "chal",
        },
    )
    event = normalize_feedback("r1", req)
    assert event.normalized_payload["winner_candidate_id"] == "chal"
    assert event.normalized_payload["kept_incumbent"] is False


# --- updater ---

def test_challenger_accepted_moves_z_toward_challenger():
    session = make_session(z=[0.0, 0.0, 0.0])
    incumbent = make_candidate("inc", z=[-0.2, -0.2, -0.2], sampler_role="incumbent")
    challenger = make_candidate("chal", z=[0.4, 0.4, 0.4], sampler_role="explorer")
    feedback = make_feedback_event("chal", "inc", "chal", kept_incumbent=False)

    updater = BestVsIncumbentUpdater()
    new_z, summary = updater.update(session, [incumbent, challenger], feedback)

    assert summary["method"] == "challenger_accepted"
    assert summary["kept_incumbent"] is False
    assert new_z[0] > 0.0


def test_incumbent_retained_does_not_move_z():
    session = make_session(z=[0.1, 0.1, 0.1])
    incumbent = make_candidate("inc", z=[-0.2, -0.2, -0.2], sampler_role="incumbent")
    challenger = make_candidate("chal", z=[0.4, 0.4, 0.4], sampler_role="explorer")
    feedback = make_feedback_event("inc", "inc", "chal", kept_incumbent=True)

    updater = BestVsIncumbentUpdater()
    new_z, summary = updater.update(session, [incumbent, challenger], feedback)

    assert summary["method"] == "incumbent_retained"
    assert new_z == [0.1, 0.1, 0.1]


def test_challenger_accepted_different_z_from_winner_average():
    """Proves best_vs_incumbent produces a different z than winner_average on same data."""
    session = make_session(z=[0.0, 0.0, 0.0])
    incumbent = make_candidate("inc", z=[-0.3, 0.0, 0.0], sampler_role="incumbent")
    challenger = make_candidate("chal", z=[0.3, 0.0, 0.0], sampler_role="explorer")
    feedback = make_feedback_event("chal", "inc", "chal", kept_incumbent=False)

    bvi_updater = BestVsIncumbentUpdater()
    bvi_z, _ = bvi_updater.update(session, [incumbent, challenger], feedback)

    wa_feedback = FeedbackEvent(
        round_id="r1",
        type=FeedbackType.winner_only,
        payload={},
        normalized_payload={"winner_candidate_id": "chal"},
    )
    wa_z, _ = WinnerAverageUpdater().update(session, [incumbent, challenger], wa_feedback)

    # bvi explicitly uses direction = challenger - incumbent; wa uses z_winner directly
    assert bvi_z != wa_z
