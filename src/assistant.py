from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from .document_loader import Chunk, chunk_documents, load_documents, supported_files
from .progress import ProgressStore
from .retriever import SemanticRetriever
from .settings import settings


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


@dataclass(frozen=True)
class BuildResult:
    document_count: int
    chunk_count: int


@dataclass(frozen=True)
class IndexStatus:
    state: str
    message: str


class LocalStudyAssistant:
    def __init__(self, data_dir: Path, index_dir: Path) -> None:
        self.data_dir = data_dir
        self.retriever = SemanticRetriever(index_dir)
        self.progress = ProgressStore(index_dir / "study_progress.sqlite3")
        self.retriever.load()

    @property
    def index_ready(self) -> bool:
        return self.retriever.ready

    @property
    def chunk_count(self) -> int:
        return len(self.retriever.chunks)

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

    def build_index(self) -> BuildResult:
        documents = load_documents(self.data_dir)
        chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
        self.retriever.build(chunks)
        (self.retriever.index_dir / "index_manifest.json").write_text(self._manifest(), encoding="utf-8")
        return BuildResult(document_count=len(documents), chunk_count=len(chunks))

    def search(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self.retriever.adaptive_search(question, top_k)

    def answer(self, question: str) -> tuple[str, list[tuple[Chunk, float]]]:
        if not self.ready_for_retrieval:
            return "자료가 변경되었거나 인덱스가 없습니다. 사이드바에서 인덱스를 갱신해주세요.", []
        sources = self.search(question)
        if not sources:
            return "먼저 사이드바에서 자료를 인덱싱해주세요.", []
        if not self._evidence_sufficient():
            return "검색된 자료가 질문을 뒷받침하기에 충분하지 않습니다. 질문을 더 구체화하거나 자료를 추가해주세요.", sources
        context = self._evidence_context(sources, settings.answer_context_characters)
        prompt = build_answer_prompt(question, context)
        return self._generate(prompt, max_tokens=420), sources

    def make_quiz(self, topic: str) -> tuple[str, list[tuple[Chunk, float]]]:
        if not self.ready_for_retrieval:
            return "자료가 변경되었거나 인덱스가 없습니다. 사이드바에서 인덱스를 갱신해주세요.", []
        # Two focused passages are sufficient for a three-question check and
        # avoid making the local model read unnecessary context on every quiz.
        sources = self.search(topic, top_k=settings.quiz_context_chunks)
        if not sources:
            return "먼저 사이드바에서 자료를 인덱싱해주세요.", []
        if not self._evidence_sufficient():
            return "검색된 자료가 퀴즈를 만들기에 충분하지 않습니다. 다른 주제로 질문하거나 자료를 추가해주세요.", sources
        context = self._evidence_context(sources, settings.quiz_context_characters)
        prompt = f"""아래 근거 문서에 명시된 내용만 사용해 '{topic}' 확인 퀴즈를 한국어로 만들어주세요.
객관식 3문항으로 작성하세요. 각 문항은 보기 4개, 정답, 한 문장 해설, 근거 번호를 포함합니다.
간결하게 작성하고, 문항당 해설은 한 문장만 씁니다.
근거가 불충분하면 문제를 만들지 말고 부족하다고 답하세요.

{context}
"""
        return self._generate(prompt, max_tokens=settings.quiz_max_tokens), sources

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
        import json
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

    def _generate(self, prompt: str, max_tokens: int) -> str:
        try:
            response = requests.post(
                f"{settings.ollama_host}/api/generate",
                json={
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
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json().get("response", "모델이 빈 응답을 반환했습니다.").strip()
        except requests.RequestException as error:
            return (
                f"로컬 모델 응답을 받지 못했습니다: `{error}`. "
                "Ollama 실행 상태와 GPU 상태를 확인해주세요. 검색 근거는 아래에서 확인할 수 있습니다."
            )
