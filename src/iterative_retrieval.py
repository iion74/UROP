"""Evidence-conditioned retrieval actions for multi-hop QA experiments.

The legacy graph actions rank every expanded paragraph against the original
question.  That is often the wrong comparison for a bridge paragraph: the
second document can be relevant only after reading the first document.  This
module keeps the retriever and reranker frozen, but scores a candidate against
``question + first-hop evidence`` and returns explicit evidence paths.

The code is deliberately separate from the production router.  It is an
experimental action space that can be audited before any full policy is
trained or the MVP behaviour is changed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .document_loader import Chunk
from .retriever import SemanticRetriever


@dataclass(frozen=True)
class BridgeSearchConfig:
    """Fixed knobs for one evidence-conditioned bridge action."""

    seed_count: int = 4
    graph_hops: int = 1
    candidates_per_seed: int = 12
    anchor_text_characters: int = 700
    conditioned_weight: float = 0.60
    anchor_weight: float = 0.15
    title_mention_weight: float = 0.15
    semantic_weight: float = 0.10

    def validate(self) -> None:
        if self.seed_count < 1 or self.graph_hops < 1 or self.candidates_per_seed < 1:
            raise ValueError("seed, graph hop, seed별 후보 수는 1 이상이어야 합니다.")
        if self.anchor_text_characters < 1:
            raise ValueError("중간 근거 글자 수는 1 이상이어야 합니다.")
        weights = (
            self.conditioned_weight,
            self.anchor_weight,
            self.title_mention_weight,
            self.semantic_weight,
        )
        if any(weight < 0.0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("bridge path score 가중치는 음수가 아니며 합이 1이어야 합니다.")


@dataclass(frozen=True)
class BridgePath:
    """One first-hop anchor and a candidate reached from that evidence."""

    anchor: Chunk
    candidate: Chunk
    distance: int
    score: float
    conditioned_score: float
    anchor_score: float
    title_mentioned: bool
    semantic_similarity: float


@dataclass(frozen=True)
class BridgeSearchResult:
    """Ranked path candidates and their deterministic search audit."""

    paths: tuple[BridgePath, ...]
    evaluated_pairs: int
    distinct_candidates: int


@dataclass(frozen=True)
class IterativeSearchConfig:
    """Beam-search controls for evidence paths containing up to four hops."""

    max_evidence_hops: int = 4
    beam_width: int = 8
    candidates_per_state: int = 8
    retrieval_candidates_per_state: int = 8
    chain_text_characters: int = 1_200
    conditioned_weight: float = 0.70
    previous_path_weight: float = 0.15
    title_mention_weight: float = 0.07
    semantic_weight: float = 0.08

    def validate(self) -> None:
        if self.max_evidence_hops < 2:
            raise ValueError("최대 evidence hop은 2 이상이어야 합니다.")
        if (
            self.beam_width < 1
            or self.candidates_per_state < 1
            or self.retrieval_candidates_per_state < 1
        ):
            raise ValueError("beam과 state별 후보 수는 1 이상이어야 합니다.")
        if self.chain_text_characters < 1:
            raise ValueError("누적 근거 글자 수는 1 이상이어야 합니다.")
        weights = (
            self.conditioned_weight,
            self.previous_path_weight,
            self.title_mention_weight,
            self.semantic_weight,
        )
        if any(weight < 0.0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("iterative path score 가중치는 음수가 아니며 합이 1이어야 합니다.")


@dataclass(frozen=True)
class EvidenceChain:
    """One ordered, cycle-free evidence chain retained by beam search."""

    chunks: tuple[Chunk, ...]
    score: float
    conditioned_score: float
    title_mentioned: bool
    semantic_similarity: float


@dataclass(frozen=True)
class IterativeSearchResult:
    """Best chains at every reachable evidence depth."""

    chains_by_hop: Mapping[int, tuple[EvidenceChain, ...]]
    evaluated_pairs_by_hop: Mapping[int, int]


def rank_bridge_paths(
    retriever: SemanticRetriever,
    question: str,
    initial: list[tuple[Chunk, float]],
    config: BridgeSearchConfig,
) -> BridgeSearchResult:
    """Rank graph candidates with the first-hop evidence visible.

    Candidate generation remains bounded by the pre-built graph.  The only
    new model operation is a batched invocation of the already frozen
    cross-encoder.  Gold evidence, answers, and benchmark decompositions are
    never used.
    """

    config.validate()
    if not initial or not retriever.graph:
        return BridgeSearchResult((), 0, 0)

    seeds = initial[: min(config.seed_count, len(initial))]
    initial_ids = {chunk.chunk_id for chunk, _ in initial}
    pairs: list[tuple[Chunk, float, Chunk, int, bool, float]] = []

    for anchor, anchor_score in seeds:
        anchor_index = retriever.chunk_positions.get(anchor.chunk_id)
        if anchor_index is None:
            continue
        reachable = retriever._reachable_distances(anchor_index, config.graph_hops)
        candidates: list[tuple[Chunk, int, bool, float]] = []
        for candidate_index, distance in reachable.items():
            if distance == 0:
                continue
            candidate = retriever.chunks[candidate_index]
            if candidate.chunk_id in initial_ids or candidate.source == anchor.source:
                continue
            similarity = _normalised_similarity(retriever._similarity(anchor_index, candidate_index))
            mentioned = title_is_mentioned(candidate.title, anchor.text)
            candidates.append((candidate, distance, mentioned, similarity))

        # Exact title bridges are more trustworthy than semantic neighbours.
        # Similarity is only a deterministic prefilter; the cross-encoder
        # performs the actual evidence-conditioned ranking below.
        candidates.sort(
            key=lambda item: (item[2], item[3], -item[1], item[0].chunk_id),
            reverse=True,
        )
        for candidate, distance, mentioned, similarity in candidates[: config.candidates_per_seed]:
            pairs.append((anchor, float(anchor_score), candidate, distance, mentioned, similarity))

    if not pairs:
        return BridgeSearchResult((), 0, 0)

    model_inputs = [
        (
            evidence_conditioned_query(
                question,
                anchor,
                max_characters=config.anchor_text_characters,
            ),
            f"{candidate.title}\n{candidate.text}",
        )
        for anchor, _, candidate, _, _, _ in pairs
    ]
    raw_scores = retriever._reranker().predict(model_inputs, show_progress_bar=False)
    conditioned_scores = _sigmoid(raw_scores)

    best_by_candidate: dict[str, BridgePath] = {}
    for pair, conditioned_score in zip(pairs, conditioned_scores, strict=True):
        anchor, anchor_score, candidate, distance, mentioned, similarity = pair
        score = (
            config.conditioned_weight * float(conditioned_score)
            + config.anchor_weight * _clamp(anchor_score)
            + config.title_mention_weight * float(mentioned)
            + config.semantic_weight * similarity
        )
        path = BridgePath(
            anchor=anchor,
            candidate=candidate,
            distance=distance,
            score=float(score),
            conditioned_score=float(conditioned_score),
            anchor_score=float(anchor_score),
            title_mentioned=mentioned,
            semantic_similarity=similarity,
        )
        previous = best_by_candidate.get(candidate.chunk_id)
        if previous is None or _path_sort_key(path) > _path_sort_key(previous):
            best_by_candidate[candidate.chunk_id] = path

    paths = tuple(sorted(best_by_candidate.values(), key=_path_sort_key, reverse=True))
    return BridgeSearchResult(paths, len(pairs), len(best_by_candidate))


def rank_iterative_chains(
    retriever: SemanticRetriever,
    question: str,
    initial: list[tuple[Chunk, float]],
    bridge_config: BridgeSearchConfig,
    iterative_config: IterativeSearchConfig,
) -> IterativeSearchResult:
    """Build 2/3/4-hop evidence chains with an evidence-conditioned beam.

    A state contains every paragraph selected so far.  The next paragraph is
    scored against the original question *and the whole accumulated chain*.
    Gold hop counts are never supplied to the search: the evaluator enumerates
    every depth and a later policy learns when to stop.
    """

    bridge_config.validate()
    iterative_config.validate()
    if not initial:
        return IterativeSearchResult({}, {})

    # Start with Direct evidence, then repeatedly retrieve again with the
    # accumulated chain visible.  Restricting every transition to graph
    # neighbours cannot solve MuSiQue-style links such as
    # ``person -> country -> event`` because the intermediate answer is often
    # present in paragraph text but not in the next paragraph title.
    beam = tuple(
        EvidenceChain(
            chunks=(chunk,),
            score=float(score),
            conditioned_score=float(score),
            title_mentioned=False,
            semantic_similarity=1.0,
        )
        for chunk, score in initial[: min(bridge_config.seed_count, len(initial))]
    )
    beam = _select_chain_beam(beam, iterative_config.beam_width)
    evaluated: dict[int, int] = {}
    chains_by_hop: dict[int, tuple[EvidenceChain, ...]] = {}

    for evidence_hop in range(2, iterative_config.max_evidence_hops + 1):
        pairs: list[tuple[EvidenceChain, Chunk, bool, float]] = []
        for chain in beam:
            last = chain.chunks[-1]
            last_index = retriever.chunk_positions.get(last.chunk_id)
            if last_index is None:
                continue
            used_chunk_ids = {chunk.chunk_id for chunk in chain.chunks}
            used_sources = {chunk.source for chunk in chain.chunks}
            candidates_by_id: dict[str, tuple[Chunk, bool, float]] = {}

            # 1) Preserve strong Direct alternatives.  A support paragraph may
            # already be in the first candidate pool even when it is not in
            # the top-k reader context.
            for candidate, _ in initial[: iterative_config.candidates_per_state]:
                if candidate.chunk_id in used_chunk_ids or candidate.source in used_sources:
                    continue
                similarity = _max_chain_similarity(retriever, chain.chunks, candidate)
                mentioned = any(title_is_mentioned(candidate.title, chunk.text) for chunk in chain.chunks)
                candidates_by_id[candidate.chunk_id] = (candidate, mentioned, similarity)

            # 2) Keep explicit graph neighbours.  They remain valuable for
            # title bridges, but are no longer the only possible transition.
            graph_candidates: list[tuple[Chunk, bool, float]] = []
            for candidate_index in retriever.graph.get(last_index, ()):
                candidate = retriever.chunks[candidate_index]
                if candidate.chunk_id in used_chunk_ids or candidate.source in used_sources:
                    continue
                similarity = _max_chain_similarity(retriever, chain.chunks, candidate)
                mentioned = any(title_is_mentioned(candidate.title, chunk.text) for chunk in chain.chunks)
                graph_candidates.append((candidate, mentioned, similarity))
            graph_candidates.sort(
                key=lambda item: (item[1], item[2], item[0].chunk_id),
                reverse=True,
            )
            for candidate, mentioned, similarity in graph_candidates[: iterative_config.candidates_per_state]:
                candidates_by_id[candidate.chunk_id] = (candidate, mentioned, similarity)

            # 3) Re-query the whole fixed corpus with accumulated evidence.
            # This is the crucial transition for non-title intermediate
            # answers (country, occupation, date, event, etc.).
            state_query = evidence_chain_query(
                question,
                chain.chunks,
                max_characters=iterative_config.chain_text_characters,
            )
            retrieved = retriever.search(
                state_query,
                top_k=iterative_config.retrieval_candidates_per_state,
            )
            for candidate, _ in retrieved:
                if candidate.chunk_id in used_chunk_ids or candidate.source in used_sources:
                    continue
                similarity = _max_chain_similarity(retriever, chain.chunks, candidate)
                mentioned = any(title_is_mentioned(candidate.title, chunk.text) for chunk in chain.chunks)
                candidates_by_id[candidate.chunk_id] = (candidate, mentioned, similarity)

            for candidate, mentioned, similarity in candidates_by_id.values():
                pairs.append((chain, candidate, mentioned, similarity))

        evaluated[evidence_hop] = len(pairs)
        if not pairs:
            chains_by_hop[evidence_hop] = ()
            beam = ()
            continue

        model_inputs = [
            (
                evidence_chain_query(
                    question,
                    chain.chunks,
                    max_characters=iterative_config.chain_text_characters,
                ),
                f"{candidate.title}\n{candidate.text}",
            )
            for chain, candidate, _, _ in pairs
        ]
        conditioned_scores = _sigmoid(
            retriever._reranker().predict(model_inputs, show_progress_bar=False)
        )
        expanded: list[EvidenceChain] = []
        for pair, conditioned_score in zip(pairs, conditioned_scores, strict=True):
            chain, candidate, mentioned, similarity = pair
            score = (
                iterative_config.conditioned_weight * float(conditioned_score)
                + iterative_config.previous_path_weight * _clamp(chain.score)
                + iterative_config.title_mention_weight * float(mentioned)
                + iterative_config.semantic_weight * similarity
            )
            expanded.append(
                EvidenceChain(
                    chunks=(*chain.chunks, candidate),
                    score=float(score),
                    conditioned_score=float(conditioned_score),
                    title_mentioned=mentioned,
                    semantic_similarity=similarity,
                )
            )
        beam = _select_chain_beam(tuple(expanded), iterative_config.beam_width)
        chains_by_hop[evidence_hop] = beam

    return IterativeSearchResult(chains_by_hop, evaluated)


def compose_bridge_evidence(
    initial: list[tuple[Chunk, float]],
    paths: tuple[BridgePath, ...],
    *,
    top_k: int,
    max_paths: int,
) -> list[tuple[Chunk, float]]:
    """Build a reader context containing complete anchor-candidate paths.

    Paths are placed before Direct fill-ins so a fixed character budget cannot
    silently remove the bridge candidate.  Duplicate anchors/candidates are
    emitted once, and unused slots retain the strongest Direct evidence.
    """

    if top_k < 1 or max_paths < 1:
        raise ValueError("top-k와 최대 path 수는 1 이상이어야 합니다.")
    if not paths:
        return initial[:top_k]

    selected: list[tuple[Chunk, float]] = []
    seen: set[str] = set()
    chosen_paths = _select_diverse_paths(paths, max_paths=max_paths)
    for path in chosen_paths:
        for chunk, score in ((path.anchor, path.anchor_score), (path.candidate, path.score)):
            if chunk.chunk_id in seen or len(selected) >= top_k:
                continue
            selected.append((chunk, float(score)))
            seen.add(chunk.chunk_id)

    for chunk, score in initial:
        if len(selected) >= top_k:
            break
        if chunk.chunk_id not in seen:
            selected.append((chunk, float(score)))
            seen.add(chunk.chunk_id)
    return selected


def compose_iterative_evidence(
    initial: list[tuple[Chunk, float]],
    chains_by_hop: Mapping[int, tuple[EvidenceChain, ...]],
    *,
    evidence_hops: int,
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Accumulate one new terminal candidate per step without erasing prior gains.

    A literal four-document path can fill almost the whole reader budget with
    distractors when one intermediate transition is wrong.  The policy state
    should instead be monotonic: begin with Direct evidence, inject at most one
    new terminal paragraph at each expansion step, and never remove an earlier
    expansion.  The separate depth actions still let the learned policy stop
    before a harmful transition.
    """

    if evidence_hops < 2 or top_k < 1:
        raise ValueError("evidence hop은 2 이상, top-k는 1 이상이어야 합니다.")
    selected = list(initial[:top_k])
    seen_chunks = {chunk.chunk_id for chunk, _ in selected}
    seen_sources = {chunk.source for chunk, _ in selected}
    protected_chunks: set[str] = set()

    for hop in range(2, evidence_hops + 1):
        chains = chains_by_hop.get(hop, ())
        ranked_terminals = sorted(
            ((chain.chunks[-1], chain.score) for chain in chains),
            key=lambda item: (item[1], item[0].chunk_id),
            reverse=True,
        )
        candidate_pair = next(
            (
                (chunk, score)
                for chunk, score in ranked_terminals
                if chunk.chunk_id not in seen_chunks and chunk.source not in seen_sources
            ),
            None,
        )
        if candidate_pair is None:
            continue
        candidate, score = candidate_pair
        if len(selected) < top_k:
            selected.append((candidate, float(score)))
        else:
            replaceable = [
                index
                for index, (chunk, _) in enumerate(selected)
                if chunk.chunk_id not in protected_chunks
            ]
            if not replaceable:
                break
            replace_index = min(replaceable, key=lambda index: (selected[index][1], -index))
            selected[replace_index] = (candidate, float(score))
        seen_chunks = {chunk.chunk_id for chunk, _ in selected}
        seen_sources = {chunk.source for chunk, _ in selected}
        protected_chunks.add(candidate.chunk_id)
    return selected


def evidence_conditioned_query(question: str, anchor: Chunk, *, max_characters: int) -> str:
    evidence = re.sub(r"\s+", " ", anchor.text).strip()[:max_characters]
    return f"Question: {question}\nFirst-hop evidence ({anchor.title}): {evidence}\nFind the next supporting document."


def evidence_chain_query(
    question: str,
    chunks: tuple[Chunk, ...],
    *,
    max_characters: int,
) -> str:
    """Construct a bounded query containing every state in the current path."""

    per_chunk = max(80, max_characters // max(1, len(chunks)))
    sections = []
    remaining = max_characters
    for position, chunk in enumerate(chunks, start=1):
        header = f"[{position}] {chunk.title}: "
        allowance = min(per_chunk, max(0, remaining - len(header)))
        text = re.sub(r"\s+", " ", chunk.text).strip()[:allowance]
        section = f"{header}{text}"
        sections.append(section)
        remaining -= len(section)
        if remaining <= 0:
            break
    evidence = "\n".join(sections)
    return f"Question: {question}\nEvidence chain:\n{evidence}\nFind the next supporting document."


def title_is_mentioned(title: str, text: str) -> bool:
    """Return whether a document title occurs as a token-bounded span."""

    normalised_title = _normalise(title)
    if not normalised_title:
        return False
    aliases = {normalised_title, _normalise(re.sub(r"\s*\([^)]*\)", "", title))}
    haystack = f" {_normalise(text)} "
    return any(alias and f" {alias} " in haystack for alias in aliases)


def _select_diverse_paths(paths: tuple[BridgePath, ...], *, max_paths: int) -> tuple[BridgePath, ...]:
    selected: list[BridgePath] = []
    used_candidates: set[str] = set()
    used_sources: set[str] = set()
    while len(selected) < max_paths:
        available = [path for path in paths if path.candidate.chunk_id not in used_candidates]
        if not available:
            break
        best = max(
            available,
            key=lambda path: (
                path.score + (0.03 if path.anchor.source not in used_sources else 0.0),
                path.title_mentioned,
                -path.distance,
                path.candidate.chunk_id,
            ),
        )
        selected.append(best)
        used_candidates.add(best.candidate.chunk_id)
        used_sources.add(best.anchor.source)
    return tuple(selected)


def _select_chain_beam(chains: tuple[EvidenceChain, ...], beam_width: int) -> tuple[EvidenceChain, ...]:
    """Keep high-scoring chains while avoiding one terminal node monopolising the beam."""

    if not chains:
        return ()
    best_by_sequence: dict[tuple[str, ...], EvidenceChain] = {}
    for chain in chains:
        key = tuple(chunk.chunk_id for chunk in chain.chunks)
        previous = best_by_sequence.get(key)
        if previous is None or chain.score > previous.score:
            best_by_sequence[key] = chain
    ranked = sorted(
        best_by_sequence.values(),
        key=lambda chain: (
            chain.score,
            chain.title_mentioned,
            chain.conditioned_score,
            chain.chunks[-1].chunk_id,
        ),
        reverse=True,
    )
    selected: list[EvidenceChain] = []
    terminal_sources: set[str] = set()
    for chain in ranked:
        if chain.chunks[-1].source in terminal_sources:
            continue
        selected.append(chain)
        terminal_sources.add(chain.chunks[-1].source)
        if len(selected) >= beam_width:
            return tuple(selected)
    for chain in ranked:
        if chain in selected:
            continue
        selected.append(chain)
        if len(selected) >= beam_width:
            break
    return tuple(selected)


def _path_sort_key(path: BridgePath) -> tuple[float, bool, int, str]:
    return (path.score, path.title_mentioned, -path.distance, path.candidate.chunk_id)


def _sigmoid(values) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def _normalised_similarity(value: float) -> float:
    return _clamp((float(value) + 1.0) / 2.0)


def _max_chain_similarity(
    retriever: SemanticRetriever,
    chunks: tuple[Chunk, ...],
    candidate: Chunk,
) -> float:
    candidate_index = retriever.chunk_positions.get(candidate.chunk_id)
    if candidate_index is None:
        return 0.0
    values = []
    for chunk in chunks:
        chunk_index = retriever.chunk_positions.get(chunk.chunk_id)
        if chunk_index is not None:
            values.append(_normalised_similarity(retriever._similarity(chunk_index, candidate_index)))
    return max(values, default=0.0)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", text.casefold()).strip()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
