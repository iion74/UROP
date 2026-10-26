from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    # Cross-encoders score a query and a passage together.  They are slower
    # than embeddings, so they only rerank a small hybrid candidate set.
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_candidates: int = int(os.getenv("RERANKER_CANDIDATES", "20"))
    evidence_min_score: float = float(os.getenv("EVIDENCE_MIN_SCORE", "0.20"))
    # Evidence-gain graph routing.  These are deliberately explicit so a
    # benchmark experiment can sweep the same policy without editing code.
    router_seed_count: int = int(os.getenv("ROUTER_SEED_COUNT", "8"))
    graph_small_budget: int = int(os.getenv("GRAPH_SMALL_BUDGET", "3"))
    graph_large_budget: int = int(os.getenv("GRAPH_LARGE_BUDGET", "6"))
    graph_utility_threshold: float = float(os.getenv("GRAPH_UTILITY_THRESHOLD", "0.12"))
    graph_expansion_decay: float = float(os.getenv("GRAPH_EXPANSION_DECAY", "0.58"))
    # The direct retriever retains only 15 candidates during policy labelling.
    # This must be larger than that frontier so graph expansion can contribute
    # genuinely new evidence rather than reselecting direct candidates.
    graph_semantic_neighbors: int = int(os.getenv("GRAPH_SEMANTIC_NEIGHBORS", "24"))
    # Title/entity bridge edges let a retrieved paragraph point to another
    # document explicitly mentioned in its text.  They complement semantic
    # neighbours with a lightweight multi-hop relation signal.
    graph_title_bridges: bool = os.getenv("GRAPH_TITLE_BRIDGES", "true").lower() in {"1", "true", "yes", "on"}
    graph_max_title_bridges_per_chunk: int = int(os.getenv("GRAPH_MAX_TITLE_BRIDGES_PER_CHUNK", "8"))
    # Graph candidates can be relevant through a bridge even when the original
    # multi-hop question has weak lexical overlap with the second document.
    # Preserve that path evidence during final reranking and reserve a bounded
    # number of result slots so graph actions are meaningfully different.
    graph_rerank_prior_weight: float = float(os.getenv("GRAPH_RERANK_PRIOR_WEIGHT", "0.35"))
    graph_source_diversity_bonus: float = float(os.getenv("GRAPH_SOURCE_DIVERSITY_BONUS", "0.08"))
    graph_min_expanded_results: int = int(os.getenv("GRAPH_MIN_EXPANDED_RESULTS", "1"))
    # Optional learned action policy.  Leave blank to use the transparent rule
    # baseline; a HotpotQA-trained .npz policy can be enabled without reindexing.
    policy_model_path: str = os.getenv("POLICY_MODEL_PATH", "")
    policy_min_confidence: float = float(os.getenv("POLICY_MIN_CONFIDENCE", "0.45"))
    policy_min_expected_advantage: float = float(os.getenv("POLICY_MIN_EXPECTED_ADVANTAGE", "0.01"))
    chunk_size: int = 900
    chunk_overlap: int = 140
    # A quiz needs enough evidence to be grounded, but passing every retrieved
    # passage makes local generation noticeably slower.
    quiz_context_chunks: int = int(os.getenv("QUIZ_CONTEXT_CHUNKS", "2"))
    quiz_max_tokens: int = int(os.getenv("QUIZ_MAX_TOKENS", "420"))
    answer_context_characters: int = int(os.getenv("ANSWER_CONTEXT_CHARACTERS", "4800"))
    quiz_context_characters: int = int(os.getenv("QUIZ_CONTEXT_CHARACTERS", "2400"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


settings = Settings()
