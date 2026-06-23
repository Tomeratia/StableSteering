from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Callable

from app.core.config import settings
from app.core.schema import (
    Candidate,
    ConvergenceReport,
    Experiment,
    ExperimentCreate,
    FeedbackRequest,
    FeedbackResponse,
    ReplayExport,
    RenderStatus,
    Round,
    RoundResponse,
    SeedPolicy,
    Session,
    SessionCreate,
    SessionStatus,
    utc_now,
)
from app.core.logging import logger
from app.engine.convergence import evaluate_convergence
from app.core.tracing import TraceRecorder
from app.feedback.normalization import normalize_feedback
from app.samplers.axis_sweep import AxisSweepSampler
from app.samplers.annealed_shell import AnnealedShellSampler
from app.samplers.base import clamp_vector
from app.samplers.diversity_shell import DiversityShellSampler
from app.samplers.exploit_orthogonal import ExploitOrthogonalSampler
from app.samplers.incumbent_mix import IncumbentMixSampler
from app.samplers.line_search import LineSearchSampler
from app.samplers.plateau_escape import PlateauEscapeSampler
from app.samplers.quality_diversity_mix import QualityDiversityMixSampler
from app.samplers.random_local import RandomLocalSampler
from app.samplers.restart_bridge_mix import RestartBridgeMixSampler
from app.samplers.spherical_cover import SphericalCoverSampler
from app.samplers.two_scale_cover import TwoScaleCoverSampler
from app.samplers.uncertainty import UncertaintyGuidedSampler
from app.updaters.advantage_softmax_pref import AdvantageSoftmaxPreferenceUpdater
from app.storage.repository import JsonRepository
from app.updaters.contrastive_pref import ContrastivePreferenceUpdater
from app.updaters.critique_weighted_pref import CritiqueWeightedPreferenceUpdater
from app.updaters.critique_momentum_pref import CritiqueMomentumPreferenceUpdater
from app.updaters.attribute_slider_pref import AttributeSliderUpdater
from app.updaters.borda_pref import BordaPreferenceUpdater
from app.updaters.bradley_terry_pref import BradleyTerryPreferenceUpdater
from app.updaters.challenger_mixture import ChallengerMixturePreferenceUpdater
from app.updaters.linear_pref import LinearPreferenceUpdater
from app.updaters.plackett_luce_pref import PlackettLucePreferenceUpdater
from app.updaters.softmax_pref import SoftmaxPreferenceUpdater
from app.updaters.score_weighted import ScoreWeightedPreferenceUpdater
from app.updaters.winner_average import WinnerAverageUpdater
from app.updaters.winner_copy import WinnerCopyUpdater
from app.engine.generation import GenerationEngine, build_generation_engine


class Orchestrator:
    """Application service that coordinates experiments, sessions, and rounds."""

    def __init__(
        self,
        repository: JsonRepository | None = None,
        generator: GenerationEngine | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.repository = repository or JsonRepository()
        self.generator = generator or build_generation_engine()
        self.trace_recorder = trace_recorder or TraceRecorder(self.repository.traces_dir)
        self.samplers = {
            "random_local": RandomLocalSampler(),
            "exploit_orthogonal": ExploitOrthogonalSampler(),
            "uncertainty_guided": UncertaintyGuidedSampler(),
            "axis_sweep": AxisSweepSampler(),
            "incumbent_mix": IncumbentMixSampler(),
            "diversity_shell": DiversityShellSampler(),
            "line_search": LineSearchSampler(),
            "plateau_escape": PlateauEscapeSampler(),
            "annealed_shell": AnnealedShellSampler(),
            "spherical_cover": SphericalCoverSampler(),
            "two_scale_cover": TwoScaleCoverSampler(),
            "quality_diversity_mix": QualityDiversityMixSampler(),
            "restart_bridge_mix": RestartBridgeMixSampler(),
        }
        self.updaters = {
            "winner_copy": WinnerCopyUpdater(),
            "winner_average": WinnerAverageUpdater(),
            "linear_preference": LinearPreferenceUpdater(),
            "score_weighted_preference": ScoreWeightedPreferenceUpdater(),
            "contrastive_preference": ContrastivePreferenceUpdater(),
            "softmax_preference": SoftmaxPreferenceUpdater(),
            "borda_preference": BordaPreferenceUpdater(),
            "bradley_terry_preference": BradleyTerryPreferenceUpdater(),
            "challenger_mixture_preference": ChallengerMixturePreferenceUpdater(),
            "plackett_luce_preference": PlackettLucePreferenceUpdater(),
            "advantage_softmax_preference": AdvantageSoftmaxPreferenceUpdater(),
            "critique_weighted_preference": CritiqueWeightedPreferenceUpdater(),
            "critique_momentum_preference": CritiqueMomentumPreferenceUpdater(),
            "attribute_slider_preference": AttributeSliderUpdater(),
        }

    @staticmethod
    def _report_progress(progress_callback: Callable[[int, str], None] | None, progress: int, message: str) -> None:
        """Emit a phase-level progress update when a callback is available."""

        if progress_callback is not None:
            progress_callback(progress, message)

    def create_experiment(self, request: ExperimentCreate) -> Experiment:
        """Create and persist a reusable experiment definition."""

        experiment = Experiment(name=request.name, description=request.description, config=request.config)
        logger.info("Creating experiment %s", experiment.id)
        self.trace_recorder.append_backend(
            "experiment.created",
            {"experiment_id": experiment.id, "name": experiment.name, "sampler": experiment.config.sampler},
        )
        return self.repository.save_experiment(experiment)

    def list_experiments(self) -> list[Experiment]:
        """Return all stored experiments."""

        return self.repository.list_experiments()

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Load one experiment by identifier."""

        return self.repository.get_experiment(experiment_id)

    def create_session(self, request: SessionCreate) -> Session:
        """Create a session from an experiment or direct configuration."""

        config = request.config
        experiment_id = request.experiment_id
        if experiment_id:
            experiment = self.repository.get_experiment(experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment not found: {experiment_id}")
            config = experiment.config
        if config is None:
            raise ValueError("Session creation requires experiment_id or config")

        session = Session(
            experiment_id=experiment_id or "ad_hoc",
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_name=config.model_name,
            config=config,
            current_z=[0.0 for _ in range(config.steering_dimension)],
            status=SessionStatus.ready,
        )
        logger.info("Created session %s for experiment %s", session.id, session.experiment_id)
        self.trace_recorder.append_backend(
            "session.created",
            {"session_id": session.id, "experiment_id": session.experiment_id, "prompt": session.prompt},
        )
        saved_session = self.repository.save_session(session)
        self.generate_trace_report(saved_session.id)
        return saved_session

    def get_session(self, session_id: str) -> Session | None:
        """Load one session by identifier."""

        return self.repository.get_session(session_id)

    def list_sessions(self) -> list[Session]:
        """Return all stored sessions ordered by recent activity."""

        return self.repository.list_sessions()

    def get_session_rounds(self, session_id: str) -> list[Round]:
        """Return ordered rounds for a given session."""

        return self.repository.list_rounds_for_session(session_id)

    def generate_round(
        self,
        session_id: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> RoundResponse:
        """Propose, render, persist, and return the next round of candidates."""

        self._report_progress(progress_callback, 14, "Checking session readiness")
        session = self._assert_round_generation_allowed(session_id)
        sampler = self.samplers[session.config.sampler]
        round_index = session.current_round + 1
        self._report_progress(progress_callback, 24, "Preparing round state")
        round_obj = Round(
            session_id=session.id,
            round_index=round_index,
            incumbent_z=session.current_z,
            trust_radius=session.config.trust_radius,
            seed_policy=session.config.seed_policy,
        )
        logger.info("Generating round %s for session %s", round_index, session.id)
        self.trace_recorder.append_backend(
            "round.generation.started",
            {"session_id": session.id, "round_index": round_index, "sampler": session.config.sampler},
        )
        carried_forward = self._build_carried_forward_candidate(session)
        baseline_candidate = self._build_baseline_prompt_candidate(session)
        sampler_seed = self._seed_token(session.id, round_index, "sampler")
        sampler_session, stagnation_streak, boosted_radius = self._sampler_session_for_round(session)
        self._report_progress(progress_callback, 36, f"Sampling {session.config.candidate_count} candidate directions")
        proposed_candidates = sampler.propose(sampler_session, sampler_seed)
        if boosted_radius is not None:
            for candidate in proposed_candidates:
                candidate.generation_params["stagnation_escape_active"] = True
                candidate.generation_params["stagnation_streak"] = stagnation_streak
                candidate.generation_params["stagnation_boosted_trust_radius"] = round(boosted_radius, 4)
        proposed_candidates = self._widen_first_round_candidates(session, proposed_candidates)
        candidates = self._compose_round_candidates(
            pinned_candidate=carried_forward or baseline_candidate,
            proposed_candidates=proposed_candidates,
            candidate_count=session.config.candidate_count,
        )
        self._assign_candidate_seeds(session, round_index, candidates)
        self._report_progress(progress_callback, 52, "Rendering candidate images on the model backend")
        # Render each candidate independently so future versions can tolerate
        # partial round failures without changing the orchestration contract.
        render_progress_start = 52
        render_progress_end = 74
        total_candidates = max(1, len(candidates))
        for index, candidate in enumerate(candidates, start=1):
            progress = render_progress_start + int((render_progress_end - render_progress_start) * ((index - 1) / total_candidates))
            candidate.round_id = round_obj.id
            if candidate.generation_params.get("carried_forward") and candidate.image_path:
                self._report_progress(
                    progress_callback,
                    progress,
                    f"Using saved image {index} of {total_candidates} from the previous winning round",
                )
                candidate.render_status = RenderStatus.succeeded
                continue
            self._report_progress(
                progress_callback,
                progress,
                f"Generating image {index} of {total_candidates} on the model backend",
            )
            candidate = self.generator.render_candidate(session, candidate)
            candidate.render_status = RenderStatus.succeeded
        round_obj.candidates = candidates
        round_obj.render_status = RenderStatus.succeeded
        round_obj.latency_ms = 15 * len(candidates)
        self._report_progress(progress_callback, 76, "Saving rendered candidates and round state")
        session.current_round = round_index
        session.status = SessionStatus.awaiting_feedback
        session.updated_at = utc_now()
        self.repository.save_round(round_obj)
        self.repository.save_session(session)
        self.trace_recorder.append_backend(
            "round.generation.completed",
            {
                "session_id": session.id,
                "round_id": round_obj.id,
                "round_index": round_index,
                "candidate_count": len(round_obj.candidates),
                "candidates": [self._candidate_trace_payload(candidate) for candidate in round_obj.candidates],
            },
        )
        self._report_progress(progress_callback, 90, "Refreshing trace report and replay data")
        self.generate_trace_report(session.id)
        self._report_progress(progress_callback, 98, "Round ready for review")
        return RoundResponse(
            round_id=round_obj.id,
            candidate_metadata=round_obj.candidates,
            image_urls=[candidate.image_path or "" for candidate in round_obj.candidates],
            state_summary={
                "session_id": session.id,
                "round_index": round_index,
                "current_z": session.current_z,
            },
        )

    def submit_feedback(
        self,
        round_id: str,
        request: FeedbackRequest,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> FeedbackResponse:
        """Normalize feedback, update state, and persist the new incumbent."""

        self._report_progress(progress_callback, 14, "Checking round readiness for feedback")
        round_obj, session = self._assert_feedback_submission_allowed(round_id, request)
        self._report_progress(progress_callback, 30, "Normalizing and validating user preferences")
        feedback = normalize_feedback(round_id, request)
        self._validate_feedback_against_round(round_obj, feedback)
        updater = self.updaters[session.config.updater]
        self._report_progress(progress_callback, 52, "Updating the steering model from your feedback")
        if isinstance(updater, CritiqueMomentumPreferenceUpdater):
            past_rounds = self.repository.list_rounds_for_session(session.id)
            next_z, update_summary = updater.update(session, round_obj.candidates, feedback, past_rounds=past_rounds)
        else:
            next_z, update_summary = updater.update(session, round_obj.candidates, feedback)
        round_obj.feedback_events.append(feedback)
        round_obj.update_summary = update_summary
        session.current_z = next_z
        session.incumbent_candidate_id = update_summary["winner_candidate_id"]
        session.status = SessionStatus.ready
        session.updated_at = utc_now()
        # Persist the round first so convergence sees this round's update_summary.
        self.repository.save_round(round_obj)
        convergence = self._evaluate_session_convergence(session)
        session.converged = convergence.converged
        session.rounds_to_convergence = convergence.rounds_to_convergence
        self._report_progress(progress_callback, 72, "Saving updated session state")
        self.repository.save_session(session)
        logger.info("Applied feedback to round %s for session %s", round_obj.id, session.id)
        self.trace_recorder.append_backend(
            "round.feedback.applied",
            {
                "session_id": session.id,
                "round_id": round_obj.id,
                "feedback_type": request.feedback_type,
                "raw_feedback_payload": request.payload,
                "normalized_feedback_payload": feedback.normalized_payload,
                "critique_text": request.critique_text,
                "winner_candidate_id": update_summary["winner_candidate_id"],
                "next_incumbent_state": next_z,
            },
        )
        self.trace_recorder.append_backend(
            "round.convergence.evaluated",
            {"session_id": session.id, "round_id": round_obj.id, **convergence.model_dump(mode="json")},
        )
        self._report_progress(progress_callback, 90, "Refreshing trace report with the new preference outcome")
        self.generate_trace_report(session.id)
        self._report_progress(progress_callback, 98, "Feedback applied and next round unlocked")
        return FeedbackResponse(update_summary=update_summary, next_incumbent_state=next_z)

    def export_replay(self, session_id: str) -> dict:
        """Return a replay-ready export bundle for one session."""

        session = self._require_session(session_id)
        experiment = self.repository.get_experiment(session.experiment_id)
        rounds = self.repository.list_rounds_for_session(session.id)
        replay = ReplayExport(
            app_version=settings.app_version,
            experiment=experiment,
            session=session,
            rounds=rounds,
        )
        return replay.model_dump(mode="json")

    def get_session_convergence(self, session_id: str) -> ConvergenceReport:
        """Return the convergence report for one session.

        Raises KeyError if the session does not exist, mirroring other lookups.
        """

        session = self._require_session(session_id)
        return self._evaluate_session_convergence(session)

    def _evaluate_session_convergence(self, session: Session) -> ConvergenceReport:
        """Compute convergence from the session's persisted round history."""

        rounds = self.repository.list_rounds_for_session(session.id)
        return evaluate_convergence(session, rounds)

    def generate_trace_report(self, session_id: str):
        """Regenerate the saved HTML trace report for one session."""

        session = self._require_session(session_id)
        experiment = self.repository.get_experiment(session.experiment_id)
        rounds = self.repository.list_rounds_for_session(session.id)
        return self.trace_recorder.write_session_report(
            session=session.model_dump(mode="json"),
            experiment=experiment.model_dump(mode="json") if experiment else None,
            rounds=[round_obj.model_dump(mode="json") for round_obj in rounds],
            backend_events=self.trace_recorder.load_session_backend_events(session.id),
            frontend_events=self.trace_recorder.load_session_frontend_events(session.id),
            diagnostics=self.generator.diagnostics(),
        )

    def _require_session(self, session_id: str) -> Session:
        """Load a session or raise a lookup error."""

        session = self.repository.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def _require_round(self, round_id: str) -> Round:
        """Load a round or raise a lookup error."""

        round_obj = self.repository.get_round(round_id)
        if round_obj is None:
            raise KeyError(f"Round not found: {round_id}")
        return round_obj

    def _assert_round_generation_allowed(self, session_id: str) -> Session:
        """Validate that a session is in a state that allows generating a round."""

        session = self._require_session(session_id)
        if session.status == SessionStatus.awaiting_feedback:
            raise RuntimeError("Cannot generate a new round while feedback for the current round is still pending")
        return session

    def _assert_feedback_submission_allowed(self, round_id: str, request: FeedbackRequest | None = None) -> tuple[Round, Session]:
        """Validate that a round can currently accept feedback."""

        round_obj = self._require_round(round_id)
        session = self._require_session(round_obj.session_id)
        if request is not None and request.feedback_type != session.config.feedback_mode:
            raise ValueError(
                f"Feedback type {request.feedback_type.value} does not match session mode {session.config.feedback_mode.value}"
            )
        if session.status != SessionStatus.awaiting_feedback:
            raise RuntimeError("Session is not currently awaiting feedback for this round")
        if round_obj.feedback_events:
            raise RuntimeError("Feedback has already been submitted for this round")
        return round_obj, session

    def _validate_feedback_against_round(self, round_obj: Round, feedback) -> None:
        """Ensure feedback references candidates that exist in the target round."""

        candidate_ids = {candidate.id for candidate in round_obj.candidates}
        winner_id = feedback.normalized_payload.get("winner_candidate_id")
        if winner_id not in candidate_ids:
            raise ValueError(f"Feedback references unknown winner candidate: {winner_id}")

        loser_id = feedback.normalized_payload.get("loser_candidate_id")
        if loser_id is not None and loser_id not in candidate_ids:
            raise ValueError(f"Feedback references unknown loser candidate: {loser_id}")

        ranking = feedback.normalized_payload.get("ranking", [])
        unknown_ranked = [candidate_id for candidate_id in ranking if candidate_id not in candidate_ids]
        if unknown_ranked:
            raise ValueError(f"Feedback ranking references unknown candidates: {', '.join(unknown_ranked)}")

        approved = feedback.normalized_payload.get("approved_candidate_ids", [])
        unknown_approved = [candidate_id for candidate_id in approved if candidate_id not in candidate_ids]
        if unknown_approved:
            raise ValueError(f"Feedback approvals reference unknown candidates: {', '.join(unknown_approved)}")

        rejected = feedback.normalized_payload.get("rejected_candidate_ids", [])
        unknown_rejected = [candidate_id for candidate_id in rejected if candidate_id not in candidate_ids]
        if unknown_rejected:
            raise ValueError(f"Feedback rejections reference unknown candidates: {', '.join(unknown_rejected)}")

    @staticmethod
    def _candidate_trace_payload(candidate) -> dict:
        """Return a compact trace payload for one proposed image candidate."""

        return {
            "candidate_id": candidate.id,
            "candidate_index": candidate.candidate_index,
            "sampler_role": candidate.sampler_role,
            "seed": candidate.seed,
            "image_path": candidate.image_path,
            "z": candidate.z,
            "predicted_score": candidate.predicted_score,
            "predicted_uncertainty": candidate.predicted_uncertainty,
            "seed_policy": candidate.generation_params.get("seed_policy"),
            "seed_group": candidate.generation_params.get("seed_group"),
        }

    def _build_carried_forward_candidate(self, session: Session) -> Candidate | None:
        """Clone the prior round's winning candidate so the next round preserves it."""

        if not session.incumbent_candidate_id or session.current_round == 0:
            return None

        previous_rounds = self.repository.list_rounds_for_session(session.id)
        if not previous_rounds:
            return None

        previous_round = previous_rounds[-1]
        winner = next(
            (candidate for candidate in previous_round.candidates if candidate.id == session.incumbent_candidate_id),
            None,
        )
        if winner is None:
            logger.warning(
                "Could not find incumbent candidate %s in previous round %s for session %s",
                session.incumbent_candidate_id,
                previous_round.id,
                session.id,
            )
            return None

        generation_params = deepcopy(winner.generation_params)
        generation_params.update(
            {
                "carried_forward": True,
                "carried_forward_candidate_id": winner.id,
                "carried_forward_round_id": previous_round.id,
            }
        )
        return Candidate(
            round_id="",
            candidate_index=0,
            z=list(winner.z),
            sampler_role="incumbent",
            predicted_score=winner.predicted_score,
            predicted_uncertainty=winner.predicted_uncertainty,
            seed=winner.seed,
            generation_params=generation_params,
            image_path=winner.image_path,
            render_status=winner.render_status,
        )

    @staticmethod
    def _build_baseline_prompt_candidate(session: Session) -> Candidate | None:
        """Create the unmodified-prompt candidate for the very first round."""

        if session.current_round != 0:
            return None

        return Candidate(
            round_id="",
            candidate_index=0,
            z=[0.0 for _ in session.current_z],
            sampler_role="baseline_prompt",
            predicted_score=0.0,
            predicted_uncertainty=0.05,
            seed=0,
            generation_params={
                "image_size": session.config.image_size,
                "baseline_prompt": True,
                "steering_applied": False,
            },
        )

    @staticmethod
    def _widen_first_round_candidates(session: Session, proposed_candidates: list[Candidate]) -> list[Candidate]:
        """Slightly spread first-round exploratory candidates away from the prompt baseline."""

        if session.current_round != 0:
            return proposed_candidates

        boosted_candidates: list[Candidate] = []
        dimensions = max(1, len(session.current_z))
        boost_radius = min(max(session.config.trust_radius * 1.55, 0.34), 0.72)
        min_radius = min(max(session.config.trust_radius * 0.95, 0.24), boost_radius)
        for index, candidate in enumerate(proposed_candidates):
            if candidate.sampler_role == "exploit":
                exploit_radius = min(max(session.config.trust_radius * 0.35, 0.12), 0.24)
                boosted_z = clamp_vector(list(candidate.z), exploit_radius)
                candidate.z = boosted_z
                candidate.generation_params["first_round_diversity_boost"] = True
                candidate.generation_params["first_round_diversity_scale"] = 0.6
                candidate.generation_params["first_round_role_behavior"] = "keep_exploit_close"
                boosted_candidates.append(candidate)
                continue

            spread_direction = Orchestrator._first_round_spread_direction(index, dimensions)
            scale = 1.15 + (0.1 * index)
            blended = [
                (original * 0.35) + (spread * boost_radius)
                for original, spread in zip(candidate.z, spread_direction, strict=False)
            ]
            boosted_z = clamp_vector(blended, boost_radius)
            length = math.sqrt(sum(value * value for value in boosted_z))
            if 0.0 < length < min_radius:
                normalization = min_radius / length
                boosted_z = clamp_vector([value * normalization for value in boosted_z], boost_radius)
                length = math.sqrt(sum(value * value for value in boosted_z))
            if length == 0.0:
                axis = index % dimensions
                boosted_z = [0.0 for _ in session.current_z]
                boosted_z[axis] = min_radius
            candidate.z = boosted_z
            candidate.generation_params["first_round_diversity_boost"] = True
            candidate.generation_params["first_round_diversity_scale"] = round(scale, 3)
            candidate.generation_params["first_round_min_radius"] = round(min_radius, 3)
            candidate.generation_params["first_round_spread_direction"] = [round(value, 4) for value in spread_direction]
            boosted_candidates.append(candidate)
        return boosted_candidates

    @staticmethod
    def _first_round_spread_direction(index: int, dimensions: int) -> list[float]:
        """Build a deliberately separated first-round direction for visible diversity."""

        vector = [0.0 for _ in range(dimensions)]
        primary_axis = index % dimensions
        secondary_axis = (index + 1) % dimensions
        tertiary_axis = (index + 2) % dimensions
        primary_sign = 1.0 if index % 2 == 0 else -1.0
        secondary_sign = -1.0 if index % 3 == 1 else 1.0
        tertiary_sign = -1.0 if index % 4 >= 2 else 1.0

        vector[primary_axis] = 1.0 * primary_sign
        if dimensions > 1:
            vector[secondary_axis] += 0.55 * secondary_sign
        if dimensions > 2:
            vector[tertiary_axis] += 0.3 * tertiary_sign
        if dimensions > 3:
            extra_axis = (index + 3) % dimensions
            vector[extra_axis] += 0.22 if index % 2 == 0 else -0.22

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0.0:
            vector[0] = 1.0
            return vector
        return [value / length for value in vector]

    @staticmethod
    def _compose_round_candidates(
        *,
        pinned_candidate: Candidate | None,
        proposed_candidates: list[Candidate],
        candidate_count: int,
    ) -> list[Candidate]:
        """Build one round batch with a required leading candidate when available."""

        selected = []
        if pinned_candidate is not None:
            selected.append(pinned_candidate)
        remaining_slots = max(0, candidate_count - len(selected))
        selected.extend(proposed_candidates[:remaining_slots])
        for index, candidate in enumerate(selected):
            candidate.candidate_index = index
        return selected

    def _assign_candidate_seeds(self, session: Session, round_index: int, candidates: list[Candidate]) -> None:
        """Assign deterministic candidate seeds according to the configured policy."""

        policy = session.config.seed_policy
        round_seed = self._seed_token(session.id, round_index, "round")
        for candidate in candidates:
            if candidate.generation_params.get("carried_forward"):
                candidate.generation_params["seed_policy"] = policy.value
                candidate.generation_params["seed_group"] = "carried_forward"
                candidate.generation_params["seed_preserved"] = True
                continue

            if policy == SeedPolicy.fixed_per_round:
                candidate.seed = round_seed
                seed_group = "round_shared"
            elif policy == SeedPolicy.fixed_per_candidate:
                candidate.seed = self._seed_token(session.id, round_index, "candidate", str(candidate.candidate_index))
                seed_group = f"candidate:{candidate.candidate_index}"
            elif policy == SeedPolicy.fixed_per_candidate_role:
                role = candidate.sampler_role or "candidate"
                candidate.seed = self._seed_token(session.id, round_index, "role", role)
                seed_group = f"role:{role}"
            else:
                raise ValueError(f"Unsupported seed policy: {policy}")

            candidate.generation_params["seed_policy"] = policy.value
            candidate.generation_params["seed_group"] = seed_group
            candidate.generation_params["round_seed"] = round_seed

    def _sampler_session_for_round(self, session: Session) -> tuple[Session, int, float | None]:
        """Return the sampling view of the session, optionally widened after repeated no-change wins."""

        patience = int(session.config.stagnation_patience)
        if patience <= 0 or session.current_round <= 0:
            return session, 0, None

        streak = self._trailing_selected_image_streak(session)
        if streak < patience:
            return session, streak, None

        boosted_radius = min(1.0, session.config.trust_radius * session.config.stagnation_trust_radius_scale)
        if boosted_radius <= session.config.trust_radius + 1e-9:
            return session, streak, None

        boosted_config = session.config.model_copy(update={"trust_radius": boosted_radius})
        boosted_session = session.model_copy(deep=True)
        boosted_session.config = boosted_config
        return boosted_session, streak, boosted_radius

    def _trailing_selected_image_streak(self, session: Session) -> int:
        """Count trailing completed rounds that ended with the same selected image artifact."""

        rounds = self.repository.list_rounds_for_session(session.id)
        streak = 0
        image_key: str | None = None
        for round_obj in reversed(rounds):
            if not round_obj.update_summary:
                break
            winner_id = round_obj.update_summary.get("winner_candidate_id")
            if not winner_id:
                break
            winner = next((candidate for candidate in round_obj.candidates if candidate.id == winner_id), None)
            if winner is None:
                break
            current_key = winner.image_path or repr([round(value, 6) for value in winner.z])
            if image_key is None:
                image_key = current_key
                streak = 1
                continue
            if current_key != image_key:
                break
            streak += 1
        return streak

    @staticmethod
    def _seed_token(*parts: object) -> int:
        """Create one stable positive seed from arbitrary deterministic inputs."""

        joined = "|".join(str(part) for part in parts)
        digest = hashlib.blake2b(joined.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)
