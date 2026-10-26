from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import requests

from .answer_gain_policy import AnswerGainPolicy, StepwiseResult, StepwiseSearcher
from .curriculum import LearningUnit, module_name
from .document_loader import Chunk, chunk_documents, load_documents, supported_files
from .learning_analytics import LearningReport, SemanticUnitIndex, build_report, notes_by_unit
from .progress import ProgressStore
from .qa_evaluation import (
    EVIDENCE_CHAIN_PROMPT_VERSION,
    AnswerCache,
    FrozenOllamaReader,
    ReaderConfig,
    ollama_model_identity,
)
from .query_rewriter import LEGACY_PROMPT_VERSION, FrozenOllamaQueryRewriter, QueryRewriteCache, QueryRewriterConfig
from .retriever import SemanticRetriever
from .settings import settings


QUIZ_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "explanation": {"type": "string"},
                    "evidence": {"type": "integer", "minimum": 1},
                },
                "required": ["question", "options", "answer_index", "explanation", "evidence"],
            },
        }
    },
    "required": ["questions"],
}


def build_answer_prompt(question: str, context: str) -> str:
    """Use the validated evidence-chain discipline while keeping a useful Korean answer."""
    return f"""당신은 자격증 학습 도우미입니다. 아래 근거 문서만 바탕으로 한국어로 답변하세요.
답변하기 전에 내부적으로 다음 검증을 순서대로 수행하세요.
1. 질문을 답하는 데 필요한 사실 관계를 나눕니다.
2. 각 관계를 뒷받침하는 가장 짧은 근거 사슬을 문서 사이에서 연결합니다.
3. 동명이인·비슷한 용어·단위·날짜가 섞이지 않았는지 확인합니다.

검증 과정은 길게 출력하지 말고 최종 답변만 명확하게 작성하세요.
근거에 필수 연결 고리가 없으면 모른다고 명확히 말하고 추측하거나 외부 지식을 보태지 마세요.
사실을 제시한 문장 끝에는 사용한 근거 번호를 [근거 1]처럼 표기하세요.

질문: {question}

근거 문서:
{context}
"""


def build_quiz_prompt(topic: str, context: str) -> str:
    return f"""아래 근거 문서에 명시된 내용만 사용해 '{topic}' 확인 퀴즈를 한국어로 만드세요.
객관식 3문항을 JSON으로 작성합니다. 각 문항은 보기 4개(options), 정답 보기의 0부터 시작하는 번호(answer_index),
한 문장 해설(explanation), 사용한 근거 번호(evidence)를 포함합니다.
정답이 근거 문서로 명확히 확인되는 문항만 만들고, 오답 보기는 그럴듯하지만 근거와 어긋나게 만드세요.

{context}
"""


def build_summary_prompt(unit_name: str, context: str) -> str:
    return f"""당신은 NCS 자격증 학습 튜터입니다. 아래 근거 문서만 사용해 '{unit_name}' 단원을 복습용으로 요약하세요.
형식: 핵심 개념 3~5개를 글머리표로 쓰고, 각 항목 끝에 [근거 1]처럼 근거 번호를 표기합니다.
마지막에 '꼭 기억할 한 줄'을 덧붙이세요. 근거에 없는 내용은 쓰지 마세요.

{context}
"""


@dataclass(frozen=True)
class BuildResult:
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class IndexStatus:
    state: str
    message: str


@dataclass(frozen=True)
class QuizItem:
    question: str
    options: tuple[str, ...]
    answer_index: int
    explanation: str
    evidence: int


@dataclass(frozen=True)
class Quiz:
    topic: str
    unit_id: str | None
    items: tuple[QuizItem, ...]
    message: str = ""


def parse_quiz(raw: str, topic: str, unit_id: str | None) -> Quiz:
    """Keep only well-formed four-option items so grading never breaks."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # A connection failure arrives as plain text from ``_generate``.
        detail = str(raw).strip() if str(raw).startswith("로컬 모델") else "다시 생성해주세요."
        return Quiz(topic, unit_id, (), f"퀴즈 형식을 해석하지 못했습니다. {detail}")
    items = []
    for entry in payload.get("questions", []) if isinstance(payload, dict) else []:
        try:
            options = tuple(str(option).strip() for option in entry["options"])
            answer = int(entry["answer_index"])
            if len(options) != 4 or not 0 <= answer < 4 or not str(entry["question"]).strip():
                continue
            items.append(QuizItem(str(entry["question"]).strip(), options, answer,
                                  str(entry.get("explanation", "")).strip(), int(entry.get("evidence", 1))))
        except (KeyError, TypeError, ValueError):
            continue
    return Quiz(topic, unit_id, tuple(items), "" if items else "근거에서 퀴즈 문항을 만들지 못했습니다.")


class ChunkLevelRetriever:
    """Expose each NCS passage as its own document to the paper's search loop.

    The policy's planner skips candidates whose ``source`` is already visible.
    Benchmark sources were single paragraphs, but an NCS source is a whole
    learning-module PDF, so passage identity keeps SEARCH able to add evidence.
    """

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return [(replace(chunk, source=chunk.chunk_id), score) for chunk, score in self.retriever.search(query, top_k)]

    def original(self, chunk: Chunk) -> Chunk:
        return self.retriever.chunks[self.retriever.chunk_positions[chunk.chunk_id]]


class LocalStudyAssistant:
    def __init__(self, data_dir: Path, index_dir: Path) -> None:
        self.data_dir = data_dir
        self.retriever = SemanticRetriever(index_dir)
        self.progress = ProgressStore(index_dir / "study_progress.sqlite3")
        self.retriever.load()
        self.last_stepwise: StepwiseResult | None = None
        self._policy: AnswerGainPolicy | None = None
        self._model_identity: tuple[str, str | None] | None = None
        self._unit_index: SemanticUnitIndex | None = None
        self.search_mode = "단계별 검색 제어 (논문 정책)" if settings.search_control == "stepwise" else "그래프 라우팅"
        if settings.search_control == "stepwise":
            try:
                self._policy = AnswerGainPolicy.from_path(Path(settings.stepwise_policy_path))
            except (OSError, ValueError, KeyError) as error:
                self.search_mode = f"그래프 라우팅 (정책 불러오기 실패: {error})"

    @property
    def index_ready(self) -> bool:
        return self.retriever.ready

    @property
    def chunk_count(self) -> int:
        return len(self.retriever.chunks)

    @property
    def policy_name(self) -> str | None:
        return self._policy.name if self._policy else None

    @property
    def index_status(self) -> IndexStatus:
        if not self.index_ready:
            return IndexStatus("missing", "인덱스가 없습니다.")
        manifest_path = self.retriever.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            return IndexStatus("stale", "기존 인덱스입니다. 자료 상태를 확인하려면 한 번 갱신하세요.")
        try:
            saved = manifest_path.read_text(encoding="utf-8")
        except OSError:
            return IndexStatus("stale", "인덱스 상태를 읽을 수 없습니다.")
        return (IndexStatus("ready", "자료와 인덱스가 동기화되어 있습니다.")
                if saved == self._manifest() else IndexStatus("stale", "자료가 변경되었습니다. 인덱스를 갱신하세요."))

    @property
    def ready_for_retrieval(self) -> bool:
        return self.index_status.state == "ready"

    @property
    def retrieval_trace(self):
        return self.retriever.last_trace

    @property
    def unit_index(self) -> SemanticUnitIndex:
        if self._unit_index is None:
            self._unit_index = SemanticUnitIndex(self.retriever)
        return self._unit_index

    def build_index(self) -> BuildResult:
        documents = load_documents(self.data_dir)
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        self.retriever.build(chunks)
        (self.retriever.index_dir / "index_manifest.json").write_text(self._manifest(), encoding="utf-8")
        self._unit_index = None
        return BuildResult(document_count=len(documents), chunk_count=len(chunks))

    def search(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self.retriever.adaptive_search(question, top_k)

    def answer(self, question: str) -> tuple[str, list[tuple[Chunk, float]]]:
        self.last_stepwise = None
        if not self.ready_for_retrieval:
            return "자료가 변경되었거나 인덱스가 없습니다. 사이드바에서 인덱스를 갱신해주세요.", []
        if self._policy is not None:
            try:
                result, sources = self._stepwise_search(question)
            except RuntimeError as error:
                return f"로컬 모델 응답을 받지 못했습니다: `{error}`", []
            self.last_stepwise = result
            sufficient = bool(sources) and max(score for _, score in sources) >= settings.evidence_min_score
            searches = result.total_searches
        else:
            sources = self.search(question)
            sufficient = self._evidence_sufficient()
            searches = 1
        if not sources:
            return "먼저 사이드바에서 자료를 인덱싱해주세요.", []
        unit_id, _ = self.unit_index.unit_for_text(question)
        self.progress.save_question(question, unit_id, searches)
        if not sufficient:
            return "검색된 자료가 질문을 뒷받침하기에 충분하지 않습니다. 질문을 더 구체화하거나 자료를 추가해주세요.", sources
        context = self._evidence_context(sources, settings.answer_context_characters)
        return self._generate(build_answer_prompt(question, context), max_tokens=420), sources

    def make_quiz(self, topic: str, unit: LearningUnit | None = None) -> tuple[Quiz, list[tuple[Chunk, float]]]:
        if not self.ready_for_retrieval:
            return Quiz(topic, None, (), "자료가 변경되었거나 인덱스가 없습니다. 사이드바에서 인덱스를 갱신해주세요."), []
        sources = self._unit_sources(unit) if unit else []
        if not sources:
            # Two focused passages are sufficient for a three-question check.
            sources = self.search(topic, top_k=settings.quiz_context_chunks)
            if not self._evidence_sufficient():
                return Quiz(topic, None, (), "검색된 자료가 퀴즈를 만들기에 충분하지 않습니다. 다른 주제로 질문하거나 자료를 추가해주세요."), sources
        unit_id = unit.unit_id if unit else self.unit_index.unit_for_text(topic)[0]
        context = self._evidence_context(sources, settings.quiz_context_characters)
        raw = self._generate(build_quiz_prompt(topic, context), max_tokens=settings.quiz_max_tokens,
                             schema=QUIZ_JSON_SCHEMA)
        return parse_quiz(raw, topic, unit_id), sources

    def summarize_unit(self, unit: LearningUnit) -> tuple[str, list[tuple[Chunk, float]]]:
        """Agent action: a grounded review summary for a weak or upcoming unit."""
        if not self.ready_for_retrieval:
            return "자료가 변경되었거나 인덱스가 없습니다. 사이드바에서 인덱스를 갱신해주세요.", []
        sources = self._unit_sources(unit) or self.search(f"{module_name(unit.module_id)} {unit.name}", top_k=3)
        context = self._evidence_context(sources, settings.answer_context_characters)
        return self._generate(build_summary_prompt(unit.name, context), max_tokens=settings.summary_max_tokens), sources

    def record_quiz(self, quiz: Quiz, correct: int) -> None:
        self.progress.save_quiz(quiz.unit_id, quiz.topic, len(quiz.items), correct)

    def learning_report(self, today: date | None = None) -> LearningReport:
        study = self.progress.study_events()
        densities = self.unit_index.densities(notes_by_unit(study)) if self.unit_index.ready else {}
        return build_report(
            study,
            self.progress.question_events(),
            self.progress.quiz_events(),
            densities,
            today or date.today(),
            self.progress.exam_date(),
        )

    def _unit_sources(self, unit: LearningUnit) -> list[tuple[Chunk, float]]:
        return [(chunk, 1.0) for chunk in self.unit_index.anchors(unit.unit_id)]

    def _stepwise_search(self, question: str) -> tuple[StepwiseResult, list[tuple[Chunk, float]]]:
        if self._model_identity is None:
            self._model_identity = ollama_model_identity(settings.ollama_host, settings.ollama_model)
        model, digest = self._model_identity
        common = {
            "host": settings.ollama_host,
            "model": model,
            "model_digest": digest,
            "seed": 42,
            "temperature": 0.0,
            "keep_alive": settings.ollama_keep_alive,
        }
        index_dir = self.retriever.index_dir
        # SQLite connections are opened per request: Streamlit reruns on other threads.
        with AnswerCache(index_dir / "stepwise_reader_cache.sqlite3") as answers, \
                QueryRewriteCache(index_dir / "stepwise_query_cache.sqlite3") as queries:
            reader = FrozenOllamaReader(ReaderConfig(
                **common, max_tokens=settings.stepwise_reader_max_tokens,
                prompt_version=EVIDENCE_CHAIN_PROMPT_VERSION, thinking=False,
            ), answers)
            rewriter = FrozenOllamaQueryRewriter(QueryRewriterConfig(
                **common, max_tokens=settings.stepwise_planner_max_tokens, prompt_version=LEGACY_PROMPT_VERSION,
            ), queries)
            passages = ChunkLevelRetriever(self.retriever)
            result = StepwiseSearcher(passages, rewriter, reader, self._policy).run(question)
        sources = [(passages.original(chunk), score) for chunk, score in result.final.sources]
        return result, sources

    def _manifest(self) -> str:
        """Fast file-state fingerprint used to avoid unnecessary re-indexing."""
        records = []
        for path in supported_files(self.data_dir):
            stat = path.stat()
            records.append({
                "path": str(path.relative_to(self.data_dir)),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            })
        payload = {
            "files": records,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _evidence_context(sources: list[tuple[Chunk, float]], character_budget: int) -> str:
        """Keep source labels while fitting evidence into a predictable prompt budget."""
        sections: list[str] = []
        remaining = character_budget
        for index, (chunk, _) in enumerate(sources, start=1):
            header = f"[근거 {index}: {chunk.source}]\n"
            if remaining <= len(header):
                break
            text = chunk.text[:remaining - len(header)].rstrip()
            sections.append(f"{header}{text}")
            remaining -= len(header) + len(text) + 2
        return "\n\n".join(sections)

    def _evidence_sufficient(self) -> bool:
        trace = self.retrieval_trace
        return trace is not None and trace.evidence_sufficient

    def _generate(self, prompt: str, max_tokens: int, schema: dict | None = None) -> str:
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            # Prevent a model unload between nearby quiz attempts.
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": max_tokens,
            },
        }
        if schema is not None:
            payload["format"] = schema
        try:
            response = requests.post(f"{settings.ollama_host}/api/generate", json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "모델이 빈 응답을 반환했습니다.").strip()
        except requests.RequestException as error:
            return (
                f"로컬 모델 응답을 받지 못했습니다: `{error}`. "
                "Ollama 실행 상태와 GPU 상태를 확인해주세요. 검색 근거는 아래에서 확인할 수 있습니다."
            )
