"""Evidence-gain-aware routing for adaptive graph retrieval.

The policy intentionally looks at the *retrieved evidence set*, not an LLM's
hidden state.  This makes every routing decision reproducible and keeps the
control path inexpensive enough for a local RAG deployment.  A learned policy
can later replace :class:`QueryRouter` without changing the retriever API.
"""

from __future__ import annotations

from dataclasses import dataclass


POLICY_ACTION_IDS = (
    "vector",
    "graph_1hop_small",
    "graph_1hop_large",
    "graph_2hop_small",
    "graph_2hop_large",
)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    graph_hops: int
    budget_tier: str


ACTION_SPECS = {
    "vector": ActionSpec("vector", 0, "none"),
    "graph_1hop_small": ActionSpec("graph_1hop_small", 1, "small"),
    "graph_1hop_large": ActionSpec("graph_1hop_large", 1, "large"),
    "graph_2hop_small": ActionSpec("graph_2hop_small", 2, "small"),
    "graph_2hop_large": ActionSpec("graph_2hop_large", 2, "large"),
}


@dataclass(frozen=True)
class EvidenceProfile:
    """Signals measured after an initial hybrid-retrieval pass.

    Every value is normalised to ``[0, 1]``.  ``*_hop_gain`` estimates the
    marginal value of candidates found at that exact graph depth; it is not a
    model-confidence score.
    """

    score_concentration: float
    score_margin: float
    source_dispersion: float
    graph_connectivity: float
    redundancy: float
    one_hop_gain: float
    two_hop_gain: float

    @property
    def uncertainty(self) -> float:
        """Low score concentration and margin indicate ambiguous evidence."""
        return _clamp(0.55 * (1.0 - self.score_concentration) + 0.45 * (1.0 - self.score_margin))


@dataclass(frozen=True)
class RouteDecision:
    """A budgeted graph-expansion decision made from an evidence profile."""

    route: str
    graph_hops: int
    expansion_budget: int
    expected_gain: float
    expected_utility: float
    reasons: tuple[str, ...]
    policy_confidence: float | None = None


class QueryRouter:
    """Choose the least expensive graph action with positive evidence utility.

    The policy is deliberately small and transparent.  It first measures how
    incomplete the direct evidence appears, then weighs graph-derived evidence
    gain against the cost of one- and two-hop expansion.  The constants are
    explicit baselines for benchmark calibration, rather than hidden claims of
    optimality.
    """

    def __init__(
        self,
        small_budget: int = 3,
        large_budget: int = 6,
        utility_threshold: float = 0.12,
    ) -> None:
        self.small_budget = small_budget
        self.large_budget = large_budget
        self.utility_threshold = utility_threshold

    def decide(self, profile: EvidenceProfile) -> RouteDecision:
        evidence_gap = _clamp(
            0.50 * profile.uncertainty
            + 0.35 * profile.source_dispersion
            + 0.15 * (1.0 - profile.redundancy)
        )
        one_gain = _clamp(0.70 * profile.one_hop_gain + 0.30 * profile.graph_connectivity)
        # A second hop must add its own useful evidence; it is not selected just
        # because one-hop expansion happened to be useful.
        two_gain = _clamp(0.35 * one_gain + 0.65 * profile.two_hop_gain)

        one_utility = evidence_gap * one_gain - 0.10
        two_utility = evidence_gap * two_gain - 0.22
        best_hops, best_gain, best_utility = (1, one_gain, one_utility)
        if two_utility > one_utility:
            best_hops, best_gain, best_utility = (2, two_gain, two_utility)

        reasons = self._reasons(profile, evidence_gap, best_hops, best_utility)
        if best_utility < self.utility_threshold:
            return RouteDecision(
                route="vector",
                graph_hops=0,
                expansion_budget=0,
                expected_gain=best_gain,
                expected_utility=best_utility,
                reasons=reasons + ("gain_below_graph_cost",),
            )

        budget = self.large_budget if best_utility >= 0.24 else self.small_budget
        return RouteDecision(
            route=f"graph_{best_hops}hop",
            graph_hops=best_hops,
            expansion_budget=budget,
            expected_gain=best_gain,
            expected_utility=best_utility,
            reasons=reasons + (f"{best_hops}hop_gain_exceeds_cost",),
        )

    def decision_for_action(
        self,
        profile: EvidenceProfile,
        action_id: str,
        policy_confidence: float | None = None,
        reason: str = "learned_policy",
    ) -> RouteDecision:
        """Convert a learned discrete action into the retriever's route format."""
        if action_id not in ACTION_SPECS:
            raise ValueError(f"지원하지 않는 정책 행동입니다: {action_id}")
        spec = ACTION_SPECS[action_id]
        if spec.graph_hops == 0:
            return RouteDecision(
                route="vector",
                graph_hops=0,
                expansion_budget=0,
                expected_gain=0.0,
                expected_utility=0.0,
                reasons=(reason, action_id),
                policy_confidence=policy_confidence,
            )

        one_gain = _clamp(0.70 * profile.one_hop_gain + 0.30 * profile.graph_connectivity)
        expected_gain = one_gain if spec.graph_hops == 1 else _clamp(0.35 * one_gain + 0.65 * profile.two_hop_gain)
        budget = self.small_budget if spec.budget_tier == "small" else self.large_budget
        return RouteDecision(
            route=f"graph_{spec.graph_hops}hop",
            graph_hops=spec.graph_hops,
            expansion_budget=budget,
            expected_gain=expected_gain,
            # The learned policy has no test-time oracle utility.  Its
            # probability is exposed separately as policy_confidence.
            expected_utility=0.0,
            reasons=(reason, action_id),
            policy_confidence=policy_confidence,
        )

    def action_id_for(self, decision: RouteDecision) -> str:
        """Map a decision back to a stable training/evaluation action ID."""
        if decision.graph_hops == 0:
            return "vector"
        tier = "small" if decision.expansion_budget <= self.small_budget else "large"
        action_id = f"graph_{decision.graph_hops}hop_{tier}"
        if action_id not in ACTION_SPECS:
            raise ValueError(f"정책 행동으로 변환할 수 없는 경로입니다: {decision}")
        return action_id

    @staticmethod
    def _reasons(
        profile: EvidenceProfile,
        evidence_gap: float,
        hops: int,
        utility: float,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if profile.uncertainty >= 0.55:
            reasons.append("retrieval_uncertainty")
        if profile.source_dispersion >= 0.45:
            reasons.append("dispersed_evidence")
        if profile.graph_connectivity >= 0.20:
            reasons.append("graph_bridge_available")
        if profile.redundancy >= 0.65:
            reasons.append("redundant_direct_evidence")
        if evidence_gap < 0.35:
            reasons.append("direct_evidence_concentrated")
        if hops == 2 and utility >= 0.12:
            reasons.append("second_hop_has_marginal_gain")
        return tuple(reasons) or ("direct_lookup",)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
