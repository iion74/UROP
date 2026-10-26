from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .document_loader import Chunk
from .policy import LearnedActionPolicy
from .routing import EvidenceProfile, QueryRouter, RouteDecision
from .settings import settings


GRAPH_FILENAME = "chunk_graph.json"
GRAPH_METADATA_FILENAME = "chunk_graph_meta.json"
GRAPH_FORMAT_VERSION = 5
TITLE_MENTION_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+(?:[A-Z][A-Za-z0-9'’.-]*|of|the|and|for|in|on|at|from|to)){0,7}"
)


@dataclass(frozen=True)
class RetrievalTrace:
    """Inspectable record of one adaptive-retrieval decision."""

    route: str
    graph_hops: int
    expansion_budget: int
    reasons: tuple[str, ...]
    initial_candidate_count: int
    reranked_candidate_count: int
    evidence_score: float
    evidence_sufficient: bool
    expected_gain: float
    expected_utility: float
    profile: EvidenceProfile
    policy_confidence: float | None


@dataclass
class _GraphCandidate:
    """Internal graph candidate with provenance from one or more seed chunks."""

    min_distance: int
    propagated_score: float
    supporters: set[int] = field(default_factory=set)


class SemanticRetriever:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.model: SentenceTransformer | None = None
        self.reranker: CrossEncoder | None = None
        self.bm25: BM25Okapi | None = None
        self.graph: dict[int, list[int]] = {}
        self.graph_title_bridge_edges = 0
        self.chunk_positions: dict[str, int] = {}
        self.rule_router = QueryRouter(
            small_budget=settings.graph_small_budget,
            large_budget=settings.graph_large_budget,
            utility_threshold=settings.graph_utility_threshold,
        )
        self.router = self.rule_router
        self.policy_mode = "규칙 기반"
        self._load_learned_policy()
        self.last_trace: RetrievalTrace | None = None

    @property
    def ready(self) -> bool:
        return self.index is not None and bool(self.chunks)

    def _load_learned_policy(self) -> None:
        """Opt in to a trained action policy without making the MVP brittle."""
        if not settings.policy_model_path:
            return
        path = Path(settings.policy_model_path)
        try:
            self.router = LearnedActionPolicy.from_path(
                path,
                self.rule_router,
                min_confidence=settings.policy_min_confidence,
                min_expected_advantage=settings.policy_min_expected_advantage,
            )
            self.policy_mode = f"학습 정책 · {path.name}"
        except (OSError, ValueError):
            # A stale or partially copied artifact must never block the study
            # assistant; its documented rule policy remains the safe fallback.
            self.router = self.rule_router
            self.policy_mode = "규칙 기반 (학습 정책 불러오기 실패)"

    def _model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(settings.embedding_model)
        return self.model

    def _reranker(self) -> CrossEncoder:
        if self.reranker is None:
            self.reranker = CrossEncoder(settings.reranker_model)
        return self.reranker

    def build(
        self,
        chunks: list[Chunk],
        *,
        show_embedding_progress: bool = False,
        persist: bool = True,
        semantic_neighbors: int | None = None,
    ) -> None:
        """Build a retrieval index.

        ``persist=False`` is used by benchmark-context experiments that build
        one small, question-local index at a time.  It avoids creating hundreds
        of thousands of short-lived FAISS/JSON files while keeping the exact
        same retrieval and graph implementation as the persistent MVP index.
        """
        if not chunks:
            raise ValueError("인덱싱할 텍스트가 없습니다.")
        texts = [f"passage: {chunk.text}" for chunk in chunks]
        vectors = self._model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_embedding_progress,
        )
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks
        self.chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
        self._build_bm25()
        self._build_graph(semantic_neighbors=semantic_neighbors)
        if persist:
            self._save()
            self._save_graph()

    def load(self) -> bool:
        index_path = self.index_dir / "vectors.faiss"
        metadata_path = self.index_dir / "chunks.json"
        if not index_path.exists() or not metadata_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        raw_chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in raw_chunks]
        self.chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        self._build_bm25()
        self._load_or_build_graph()
        return True

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Hybrid BM25+dense retrieval followed by a cross-encoder reranker."""
        if not self.ready or self.bm25 is None:
            return []
        vector = self._model().encode([f"query: {query}"], normalize_embeddings=True)
        candidate_count = min(max(top_k, settings.reranker_candidates), len(self.chunks))
        _, dense_indices = self.index.search(np.asarray(vector, dtype="float32"), candidate_count)
        dense_rank = {index: rank for rank, index in enumerate(dense_indices[0], start=1) if index >= 0}
        sparse_indices = np.argsort(self.bm25.get_scores(_tokenize(query)))[::-1][:candidate_count]
        sparse_rank = {int(index): rank for rank, index in enumerate(sparse_indices, start=1)}
        candidates = set(dense_rank) | set(sparse_rank)
        fused = sorted(
            candidates,
            key=lambda index: 1 / (60 + dense_rank.get(index, 10000)) + 1 / (60 + sparse_rank.get(index, 10000)),
            reverse=True,
        )
        hybrid = [
            (
                self.chunks[index],
                1 / (60 + dense_rank.get(index, 10000)) + 1 / (60 + sparse_rank.get(index, 10000)),
            )
            for index in fused
        ]
        return self._rerank(query, hybrid)[:top_k]

    def adaptive_search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Adapt graph depth and budget using the first-pass evidence profile."""
        initial = self.search(query, top_k=max(top_k * 3, 12))
        profile = self._evidence_profile(initial)
        decision = self.router.decide(profile)
        if decision.graph_hops and not self.graph:
            decision = RouteDecision(
                route="vector",
                graph_hops=0,
                expansion_budget=0,
                expected_gain=decision.expected_gain,
                expected_utility=decision.expected_utility,
                reasons=decision.reasons + ("graph_unavailable",),
            )
        if decision.graph_hops == 0:
            result = initial[:top_k]
            self.last_trace = self._trace(decision, profile, result, len(initial), len(initial))
            return result

        expanded = self._expand_for_plan(initial, decision)
        if len(expanded) == len(initial):
            decision = RouteDecision(
                route="vector",
                graph_hops=0,
                expansion_budget=0,
                expected_gain=decision.expected_gain,
                expected_utility=decision.expected_utility,
                reasons=decision.reasons + ("no_new_graph_evidence",),
            )
            result = initial[:top_k]
            self.last_trace = self._trace(decision, profile, result, len(initial), len(initial))
            return result
        reranked = self._rerank_graph_expansion(query, initial, expanded, top_k)
        self.last_trace = self._trace(decision, profile, reranked, len(initial), len(expanded))
        return reranked

    def _rerank(self, query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        raw_scores = self._reranker().predict([(query, chunk.text) for chunk, _ in candidates], show_progress_bar=False)
        # CrossEncoder scores are logits for this model.  A sigmoid yields an
        # interpretable 0-1 relevance confidence for UI and abstention checks.
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype="float64")))
        ranked = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _rerank_graph_expansion(
        self,
        query: str,
        initial: list[tuple[Chunk, float]],
        expanded: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Rerank graph evidence without erasing the bridge that found it.

        A second-hop paragraph often has little direct lexical overlap with the
        original question.  Pure cross-encoder reranking therefore discarded
        graph candidates before the policy could benefit from them.  This
        method blends the bounded graph-path gain with query relevance, adds a
        small new-source bonus, and reserves at most the configured number of
        slots for the best expanded evidence.  A bad expansion can still lower
        the oracle action utility, so the learned policy remains responsible
        for deciding when the action is worthwhile.
        """
        if not expanded or top_k < 1:
            return []
        initial_ids = {chunk.chunk_id for chunk, _ in initial}
        raw_scores = self._reranker().predict(
            [(query, chunk.text) for chunk, _ in expanded],
            show_progress_bar=False,
        )
        query_scores = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype="float64")))
        prior_weight = _clamp(settings.graph_rerank_prior_weight)
        ranked: list[tuple[Chunk, float, bool]] = []
        for (chunk, prior), query_score in zip(expanded, query_scores, strict=True):
            is_graph = chunk.chunk_id not in initial_ids
            combined = float(query_score)
            if is_graph:
                combined = (1.0 - prior_weight) * combined + prior_weight * _clamp(float(prior))
            ranked.append((chunk, combined, is_graph))

        selected: list[tuple[Chunk, float, bool]] = []
        remaining = list(ranked)
        represented_sources: set[str] = set()
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda item: item[1]
                + (settings.graph_source_diversity_bonus if item[0].source not in represented_sources else 0.0),
            )
            selected.append(best)
            represented_sources.add(best[0].source)
            remaining.remove(best)

        graph_ranked = sorted((item for item in ranked if item[2]), key=lambda item: item[1], reverse=True)
        reserved = min(max(0, settings.graph_min_expanded_results), len(graph_ranked), top_k)
        selected_graph_ids = {item[0].chunk_id for item in selected if item[2]}
        needed = max(0, reserved - len(selected_graph_ids))
        missing = [item for item in graph_ranked if item[0].chunk_id not in selected_graph_ids][:needed]
        for graph_item in missing:
            replaceable = [item for item in selected if not item[2]]
            if not replaceable:
                break
            selected.remove(min(replaceable, key=lambda item: item[1]))
            selected.append(graph_item)
        return [(chunk, score) for chunk, score, _ in sorted(selected, key=lambda item: item[1], reverse=True)]

    def _trace(
        self,
        decision: RouteDecision,
        profile: EvidenceProfile,
        sources: list[tuple[Chunk, float]],
        initial_candidate_count: int,
        reranked_candidate_count: int,
    ) -> RetrievalTrace:
        evidence_score = sources[0][1] if sources else 0.0
        return RetrievalTrace(
            route=decision.route,
            graph_hops=decision.graph_hops,
            expansion_budget=decision.expansion_budget,
            reasons=decision.reasons,
            initial_candidate_count=initial_candidate_count,
            reranked_candidate_count=reranked_candidate_count,
            evidence_score=evidence_score,
            evidence_sufficient=bool(sources) and evidence_score >= settings.evidence_min_score,
            expected_gain=decision.expected_gain,
            expected_utility=decision.expected_utility,
            profile=profile,
            policy_confidence=decision.policy_confidence,
        )

    def _evidence_profile(self, initial: list[tuple[Chunk, float]]) -> EvidenceProfile:
        """Measure uncertainty, diversity, and graph expansion value in one pass."""
        seeds = initial[: min(settings.router_seed_count, len(initial))]
        if not seeds:
            return EvidenceProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        seed_indices = [self.chunk_positions[chunk.chunk_id] for chunk, _ in seeds]
        # These are exactly the direct candidates that an expansion is not
        # allowed to reselect.  Estimating graph gain with a looser exclusion
        # set would create features that promise evidence unavailable to the
        # action executor.
        direct_indices = {self.chunk_positions[chunk.chunk_id] for chunk, _ in initial}
        scores = np.asarray([max(0.0, score) for _, score in seeds], dtype="float64")
        score_total = float(scores.sum())
        top_share = float(scores[0] / score_total) if score_total else 1.0
        if len(scores) == 1:
            concentration = 1.0
            margin = 1.0
        else:
            concentration = (top_share - 1.0 / len(scores)) / (1.0 - 1.0 / len(scores))
            margin = float((scores[0] - scores[1]) / max(scores[0], 1e-8))

        source_dispersion = (len({chunk.source for chunk, _ in seeds}) - 1) / max(1, len(seeds) - 1)
        return EvidenceProfile(
            score_concentration=_clamp(concentration),
            score_margin=_clamp(margin),
            source_dispersion=_clamp(source_dispersion),
            graph_connectivity=self._graph_connectivity(seed_indices),
            redundancy=self._mean_similarity(seed_indices),
            one_hop_gain=self._estimate_hop_gain(seeds, hop=1, excluded_indices=direct_indices),
            two_hop_gain=self._estimate_hop_gain(seeds, hop=2, excluded_indices=direct_indices),
        )

    def _expand_for_plan(
        self,
        initial: list[tuple[Chunk, float]],
        decision: RouteDecision,
    ) -> list[tuple[Chunk, float]]:
        """Add only the graph candidates with the greatest marginal evidence gain."""
        seeds = initial[: min(settings.router_seed_count, len(initial))]
        direct_indices = {self.chunk_positions[chunk.chunk_id] for chunk, _ in initial}
        graph_candidates = self._graph_candidates(
            seeds,
            max_hops=decision.graph_hops,
            excluded_indices=direct_indices,
        )
        selected = self._select_expansion_candidates(
            graph_candidates,
            anchors=[self.chunk_positions[chunk.chunk_id] for chunk, _ in seeds],
            budget=decision.expansion_budget,
        )
        # Preserve every direct result and only pay extra cross-encoder cost for
        # the bounded number of graph candidates chosen by the policy.
        return initial + [(self.chunks[index], gain) for index, gain in selected]

    def _graph_candidates(
        self,
        seeds: list[tuple[Chunk, float]],
        max_hops: int,
        excluded_indices: set[int] | None = None,
    ) -> dict[int, _GraphCandidate]:
        seed_indices = {self.chunk_positions[chunk.chunk_id] for chunk, _ in seeds}
        excluded_indices = seed_indices | (excluded_indices or set())
        candidates: dict[int, _GraphCandidate] = {}
        for chunk, score in seeds:
            seed_index = self.chunk_positions[chunk.chunk_id]
            for index, distance in self._reachable_distances(seed_index, max_hops).items():
                if distance == 0 or index in excluded_indices:
                    continue
                propagated = max(0.0, score) * settings.graph_expansion_decay ** distance
                candidate = candidates.get(index)
                if candidate is None:
                    candidates[index] = _GraphCandidate(distance, propagated, {seed_index})
                    continue
                candidate.min_distance = min(candidate.min_distance, distance)
                candidate.propagated_score = max(candidate.propagated_score, propagated)
                candidate.supporters.add(seed_index)
        return candidates

    def _select_expansion_candidates(
        self,
        candidates: dict[int, _GraphCandidate],
        anchors: list[int],
        budget: int,
    ) -> list[tuple[int, float]]:
        """MMR-style selection with explicit source and bridge-coverage rewards."""
        selected: list[int] = []
        selected_with_gain: list[tuple[int, float]] = []
        represented_sources = {self.chunks[index].source for index in anchors}
        while candidates and len(selected) < budget:
            best_index, best_gain = max(
                (
                    (index, self._marginal_evidence_gain(index, candidate, anchors, selected, represented_sources))
                    for index, candidate in candidates.items()
                ),
                key=lambda item: item[1],
            )
            selected.append(best_index)
            selected_with_gain.append((best_index, best_gain))
            represented_sources.add(self.chunks[best_index].source)
            del candidates[best_index]
        return selected_with_gain

    def _marginal_evidence_gain(
        self,
        index: int,
        candidate: _GraphCandidate,
        anchors: list[int],
        selected: list[int],
        represented_sources: set[str],
    ) -> float:
        compared = anchors + selected
        redundancy = max((self._similarity(index, other) for other in compared), default=0.0)
        novelty = 1.0 - _clamp((redundancy + 1.0) / 2.0)
        source_coverage = 1.0 if self.chunks[index].source not in represented_sources else 0.40
        bridge_coverage = min(1.0, len(candidate.supporters) / 2.0)
        return (
            0.45 * _clamp(candidate.propagated_score)
            + 0.20 * novelty
            + 0.20 * source_coverage
            + 0.15 * bridge_coverage
        )

    def _estimate_hop_gain(
        self,
        seeds: list[tuple[Chunk, float]],
        hop: int,
        excluded_indices: set[int] | None = None,
    ) -> float:
        if not self.graph:
            return 0.0
        candidates = self._graph_candidates(seeds, max_hops=hop, excluded_indices=excluded_indices)
        candidates = {index: candidate for index, candidate in candidates.items() if candidate.min_distance == hop}
        if not candidates:
            return 0.0
        anchors = [self.chunk_positions[chunk.chunk_id] for chunk, _ in seeds]
        anchor_sources = {self.chunks[index].source for index in anchors}
        gains = [
            self._marginal_evidence_gain(index, candidate, anchors, [], anchor_sources)
            for index, candidate in candidates.items()
        ]
        top_gains = sorted(gains, reverse=True)[: settings.graph_large_budget]
        return float(np.mean(top_gains)) if top_gains else 0.0

    def _graph_connectivity(self, seed_indices: list[int]) -> float:
        if len(seed_indices) < 2 or not self.graph:
            return 0.0
        reachable = {index: self._reachable_distances(index, max_hops=2) for index in seed_indices}
        connected = sum(second in reachable[first] for first, second in combinations(seed_indices, 2))
        total = len(seed_indices) * (len(seed_indices) - 1) / 2
        return _clamp(connected / total) if total else 0.0

    def _reachable_distances(self, start: int, max_hops: int) -> dict[int, int]:
        distances = {start: 0}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            if distances[node] >= max_hops:
                continue
            for neighbor in self.graph.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

    def _mean_similarity(self, indices: list[int]) -> float:
        if len(indices) < 2:
            return 0.0
        similarities = [self._similarity(first, second) for first, second in combinations(indices, 2)]
        return _clamp((float(np.mean(similarities)) + 1.0) / 2.0) if similarities else 0.0

    def _similarity(self, first: int, second: int) -> float:
        if self.index is None:
            return 0.0
        return float(np.dot(self.index.reconstruct(first), self.index.reconstruct(second)))

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "vectors.faiss"))
        (self.index_dir / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_or_build_graph(self) -> None:
        graph_path = self.index_dir / GRAPH_FILENAME
        metadata_path = self.index_dir / GRAPH_METADATA_FILENAME
        if graph_path.exists() and self._matches_graph_metadata(metadata_path):
            try:
                raw_graph = json.loads(graph_path.read_text(encoding="utf-8"))
                self.graph = {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in raw_graph.items()}
                if len(self.graph) == len(self.chunks):
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    self.graph_title_bridge_edges = int(metadata.get("title_bridge_edges", 0))
                    return
            except (OSError, ValueError, TypeError):
                pass
        self._build_graph()
        self._save_graph()

    def _matches_graph_metadata(self, path: Path) -> bool:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            return (
                metadata.get("format_version") == GRAPH_FORMAT_VERSION
                and metadata.get("semantic_neighbors") == settings.graph_semantic_neighbors
                and metadata.get("title_bridges") == settings.graph_title_bridges
                and metadata.get("max_title_bridges_per_chunk") == settings.graph_max_title_bridges_per_chunk
                and metadata.get("chunk_count") == len(self.chunks)
            )
        except (OSError, ValueError, TypeError):
            return False

    def _build_graph(self, semantic_neighbors: int | None = None) -> None:
        """Create a local graph from adjacency, semantic, and title-bridge edges."""
        semantic_neighbors = max(1, semantic_neighbors or settings.graph_semantic_neighbors)
        graph_sets = {index: set() for index in range(len(self.chunks))}
        previous_by_source: dict[str, int] = {}
        for index, chunk in enumerate(self.chunks):
            previous = previous_by_source.get(chunk.source)
            if previous is not None:
                graph_sets[index].add(previous)
                graph_sets[previous].add(index)
            previous_by_source[chunk.source] = index

        if self.index is not None and self.chunks:
            vectors = np.asarray([self.index.reconstruct(index) for index in range(len(self.chunks))], dtype="float32")
            _, neighbors = self.index.search(vectors, min(semantic_neighbors + 1, len(self.chunks)))
            for index, row in enumerate(neighbors):
                for neighbor in row:
                    neighbor = int(neighbor)
                    if neighbor >= 0 and neighbor != index:
                        graph_sets[index].add(neighbor)
                        graph_sets[neighbor].add(index)
        self.graph_title_bridge_edges = self._add_title_bridge_edges(graph_sets) if settings.graph_title_bridges else 0
        self.graph = {index: sorted(neighbors) for index, neighbors in graph_sets.items()}

    def _add_title_bridge_edges(self, graph_sets: dict[int, set[int]]) -> int:
        """Connect a chunk to documents whose titles it explicitly mentions.

        HotpotQA multi-hop evidence often moves from one article to another via
        an article name in the first paragraph.  This deterministic matcher is
        deliberately lightweight: it uses no LLM or external NER model and is
        also safe to disable for ablations.
        """
        alias_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            for alias in _source_title_aliases(chunk.title):
                alias_to_indices[alias].append(index)

        aliases_by_first_token: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
        for alias in alias_to_indices:
            tokens = tuple(alias.split())
            if tokens:
                aliases_by_first_token[tokens[0]].append((tokens, alias))
        for aliases in aliases_by_first_token.values():
            aliases.sort(key=lambda item: (-len(item[0]), item[1]))

        bridges = 0
        limit = max(1, settings.graph_max_title_bridges_per_chunk)
        for index, chunk in enumerate(self.chunks):
            added_for_chunk = 0
            mentions = _matched_title_aliases(chunk.text, aliases_by_first_token)
            for mention in sorted(mentions, key=lambda value: (-len(value.split()), value)):
                best_chunk_by_source: dict[str, tuple[int, float]] = {}
                for target in alias_to_indices.get(mention, []):
                    target_source = self.chunks[target].source
                    if target == index or target_source == chunk.source:
                        continue
                    similarity = self._similarity(index, target)
                    previous = best_chunk_by_source.get(target_source)
                    if previous is None or similarity > previous[1]:
                        best_chunk_by_source[target_source] = (target, similarity)
                targets = [
                    target
                    for target, _ in sorted(best_chunk_by_source.values(), key=lambda item: item[1], reverse=True)
                ]
                for target in targets:
                    if target not in graph_sets[index]:
                        graph_sets[index].add(target)
                        graph_sets[target].add(index)
                        bridges += 1
                        added_for_chunk += 1
                    if added_for_chunk >= limit:
                        break
                if added_for_chunk >= limit:
                    break
        return bridges

    def _save_graph(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / GRAPH_FILENAME).write_text(json.dumps(self.graph, ensure_ascii=False), encoding="utf-8")
        (self.index_dir / GRAPH_METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "format_version": GRAPH_FORMAT_VERSION,
                    "semantic_neighbors": settings.graph_semantic_neighbors,
                    "title_bridges": settings.graph_title_bridges,
                    "max_title_bridges_per_chunk": settings.graph_max_title_bridges_per_chunk,
                    "title_bridge_edges": self.graph_title_bridge_edges,
                    "chunk_count": len(self.chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _build_bm25(self) -> None:
        self.bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in self.chunks])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def _source_title_aliases(source: str) -> tuple[str, ...]:
    aliases = {_normalise_bridge_text(source)}
    aliases.add(_normalise_bridge_text(re.sub(r"\s*\([^)]*\)", "", source)))
    aliases.discard("")
    return tuple(alias for alias in aliases if len(alias) >= 3)


def _title_mentions(text: str) -> set[str]:
    return {
        normalised
        for match in TITLE_MENTION_PATTERN.findall(text)
        if len(normalised := _normalise_bridge_text(match)) >= 3
    }


def _matched_title_aliases(
    text: str,
    aliases_by_first_token: dict[str, list[tuple[tuple[str, ...], str]]],
) -> set[str]:
    """Find exact normalised title spans, including lower-case mentions.

    The older capitalised-phrase regex could miss a correct bridge or capture
    a phrase longer than the actual title.  A first-token index keeps direct
    span matching efficient without an external entity linker.
    """
    tokens = tuple(_normalise_bridge_text(text).split())
    matches = set(_title_mentions(text))
    for position, token in enumerate(tokens):
        for alias_tokens, alias in aliases_by_first_token.get(token, []):
            end = position + len(alias_tokens)
            if tokens[position:end] == alias_tokens:
                matches.add(alias)
    return matches


def _normalise_bridge_text(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", text.casefold()).strip()
