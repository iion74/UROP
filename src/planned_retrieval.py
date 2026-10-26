"""Sequential retrieval driven by a frozen next-query reformulator.

Depth is not a prediction target.  A policy repeatedly observes the current
search state and chooses STOP or CONTINUE.  ``max_search_steps`` is only a
safety/cost guard, so the same executor can run for 4, 10, or more transitions
without adding new action classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .document_loader import Chunk
from .query_rewriter import FrozenOllamaQueryRewriter, QueryRewriteResult
from .retriever import SemanticRetriever


@dataclass(frozen=True)
class PlannedRetrievalStep:
    search_step: int
    decision: "StopDecision"
    state: "SequentialSearchState"
    rewrite: QueryRewriteResult
    added_chunk: Chunk | None
    added_score: float | None
    sources: tuple[tuple[Chunk, float], ...]
    added_chunks: tuple[Chunk, ...] = ()
    added_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class PlannedRetrievalResult:
    sources_by_step: dict[int, tuple[tuple[Chunk, float], ...]]
    memory: tuple[tuple[Chunk, float], ...]
    steps: tuple[PlannedRetrievalStep, ...]
    stop_step: int
    stopped_by_policy: bool


@dataclass(frozen=True)
class SequentialSearchState:
    """Policy-visible state; it deliberately contains no gold hop or answer."""

    question: str
    search_step: int
    reader_sources: tuple[tuple[Chunk, float], ...]
    memory: tuple[tuple[Chunk, float], ...]
    previous_queries: tuple[str, ...]
    last_added_score: float | None
    consecutive_no_new: int
    evidence_sufficient: bool | None
    missing_information: str


@dataclass(frozen=True)
class StopDecision:
    continue_search: bool
    confidence: float
    reason: str


class StopPolicy(Protocol):
    def decide(self, state: SequentialSearchState) -> StopDecision: ...


class AlwaysContinuePolicy:
    """Trajectory-enumeration policy used only to create offline labels."""

    def decide(self, state: SequentialSearchState) -> StopDecision:
        del state
        return StopDecision(True, 1.0, "enumerate_offline_trajectory")


def run_planned_retrieval(
    retriever: SemanticRetriever,
    rewriter: FrozenOllamaQueryRewriter,
    question: str,
    initial: list[tuple[Chunk, float]],
    *,
    max_search_steps: int = 3,
    top_k: int = 5,
    search_candidates: int = 8,
    context_characters: int = 4_000,
    anchor_slots: int = 2,
    reader_source_cap: int | None = None,
    additions_per_step: int = 1,
    replace_when_full: bool = True,
    stop_policy: StopPolicy | None = None,
) -> PlannedRetrievalResult:
    """Repeat STOP/CONTINUE decisions until policy stop or a safety cap.

    Offline Oracle generation passes ``AlwaysContinuePolicy`` (the default)
    and scores every prefix.  Deployment passes a learned stop policy.  Gold
    hop counts are never an argument to either path.
    """

    source_cap = reader_source_cap if reader_source_cap is not None else top_k
    if (
        max_search_steps < 1
        or top_k < 1
        or search_candidates < 1
        or context_characters < 1
        or additions_per_step < 1
        or not 0 <= anchor_slots <= top_k
        or source_cap < top_k
    ):
        raise ValueError("search step/top-k/search candidate/context 값은 양수여야 합니다.")
    policy = stop_policy or AlwaysContinuePolicy()
    selected = list(initial[:top_k])
    memory = list(initial)
    memory_ids = {chunk.chunk_id for chunk, _ in memory}
    # Keep a small number of original-question anchors alive.  Previously each
    # rewrite could evict a necessary first-hop fact, which made support recall
    # fall again at later steps even though the full memory still contained it.
    anchor_ids: set[str] = {
        chunk.chunk_id for chunk, _ in selected[: min(anchor_slots, len(selected))]
    }
    protected: set[str] = set(anchor_ids)
    previous_queries: list[str] = []
    surviving_additions: list[str] = []
    sources_by_step: dict[int, tuple[tuple[Chunk, float], ...]] = {0: tuple(selected)}
    steps: list[PlannedRetrievalStep] = []
    last_added_score: float | None = None
    consecutive_no_new = 0
    stopped_by_policy = False

    for search_step in range(1, max_search_steps + 1):
        evidence_sources = _memory_context_sources(selected, memory)
        evidence = _planner_context(evidence_sources, context_characters)
        rewrite = rewriter.rewrite(question, evidence, tuple(previous_queries))
        state = SequentialSearchState(
            question=question,
            search_step=search_step - 1,
            reader_sources=tuple(selected),
            memory=tuple(memory),
            previous_queries=tuple(previous_queries),
            last_added_score=last_added_score,
            consecutive_no_new=consecutive_no_new,
            evidence_sufficient=rewrite.sufficient,
            missing_information=rewrite.missing_information,
        )
        decision = policy.decide(state)
        if not decision.continue_search:
            stopped_by_policy = True
            break

        previous_queries.append(rewrite.query)
        candidates = retriever.search(rewrite.query, top_k=search_candidates)
        for candidate, score in candidates:
            if candidate.chunk_id not in memory_ids:
                memory.append((candidate, float(score)))
                memory_ids.add(candidate.chunk_id)
        seen_chunks = {chunk.chunk_id for chunk, _ in selected}
        seen_sources = {chunk.source for chunk, _ in selected}
        additions: list[tuple[Chunk, float]] = []
        candidate_chunks = set(seen_chunks)
        candidate_sources = set(seen_sources)
        for chunk, score in candidates:
            if chunk.chunk_id in candidate_chunks or chunk.source in candidate_sources:
                continue
            if not replace_when_full and len(selected) + len(additions) >= source_cap:
                break
            additions.append((chunk, float(score)))
            candidate_chunks.add(chunk.chunk_id)
            candidate_sources.add(chunk.source)
            if len(additions) >= additions_per_step:
                break

        applied: list[tuple[Chunk, float]] = []
        for chunk, score in additions:
            if len(selected) < source_cap:
                # New gap-specific evidence must be visible before residual
                # first-pass distractors, while original anchors stay first.
                insert_at = min(len(selected), anchor_slots + len(surviving_additions))
                selected.insert(insert_at, (chunk, float(score)))
                surviving_additions.append(chunk.chunk_id)
            else:
                replaceable = [
                    index
                    for index, (current, _) in enumerate(selected)
                    if current.chunk_id not in protected
                ]
                if replaceable:
                    replace_index = min(replaceable, key=lambda index: (selected[index][1], -index))
                    selected[replace_index] = (chunk, float(score))
                    surviving_additions.append(chunk.chunk_id)
                else:
                    non_anchor_indices = [
                        index
                        for index, (current, _) in enumerate(selected)
                        if current.chunk_id not in anchor_ids
                    ]
                    replace_index = min(
                        non_anchor_indices or list(range(len(selected))),
                        key=lambda index: (selected[index][1], -index),
                    )
                    protected.discard(selected[replace_index][0].chunk_id)
                    if selected[replace_index][0].chunk_id in surviving_additions:
                        surviving_additions.remove(selected[replace_index][0].chunk_id)
                    selected[replace_index] = (chunk, float(score))
                    surviving_additions.append(chunk.chunk_id)
            protected.add(chunk.chunk_id)
            applied.append((chunk, float(score)))

        if applied:
            added_chunk, added_score = applied[0]
            last_added_score = max(score for _, score in applied)
            consecutive_no_new = 0
        else:
            added_chunk, added_score = None, None
            last_added_score = None
            consecutive_no_new += 1
        snapshot = tuple(selected)
        sources_by_step[search_step] = snapshot
        steps.append(
            PlannedRetrievalStep(
                search_step=search_step,
                decision=decision,
                state=state,
                rewrite=rewrite,
                added_chunk=added_chunk,
                added_score=added_score,
                sources=snapshot,
                added_chunks=tuple(chunk for chunk, _ in applied),
                added_scores=tuple(score for _, score in applied),
            )
        )
    return PlannedRetrievalResult(
        sources_by_step=sources_by_step,
        memory=tuple(memory),
        steps=tuple(steps),
        stop_step=len(steps),
        stopped_by_policy=stopped_by_policy,
    )


def _memory_context_sources(
    selected: list[tuple[Chunk, float]],
    memory: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """Show the planner current reader evidence followed by older memory."""

    combined = list(selected)
    seen = {chunk.chunk_id for chunk, _ in combined}
    for item in reversed(memory):
        if item[0].chunk_id not in seen:
            combined.append(item)
            seen.add(item[0].chunk_id)
    return combined


def _planner_context(sources: list[tuple[Chunk, float]], character_budget: int) -> str:
    sections: list[str] = []
    remaining = character_budget
    for position, (chunk, _) in enumerate(sources, start=1):
        header = f"[{position}] {chunk.title}\n"
        if remaining <= len(header):
            break
        text = chunk.text[: remaining - len(header)].rstrip()
        if not text:
            continue
        sections.append(f"{header}{text}")
        remaining -= len(header) + len(text) + 2
        if remaining <= 0:
            break
    return "\n\n".join(sections)
