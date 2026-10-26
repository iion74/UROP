from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningUnit:
    module_id: str
    unit_id: str
    name: str


@dataclass(frozen=True)
class Module:
    module_id: str
    name: str
    filename_prefix: str
    prerequisites: tuple[str, ...]
    units: tuple[str, ...]


MODULES: tuple[Module, ...] = (
    Module("plan", "01. 플랫폼 구축 계획", "LM2001070101", (), (
        "인공지능 플랫폼 동향 분석", "인공지능 플랫폼 목표 설정", "인공지능 플랫폼 구축 범위 정의",
        "인공지능 플랫폼 구축 일정 계획", "인공지능 플랫폼 구축 비용 계획",
    )),
    Module("requirements", "02. 요구사항 분석", "LM2001070102", ("plan",), (
        "요구사항 수집", "우선순위 도출", "요구사항 정의", "요구사항 명세화", "요구사항 점검", "요구사항 보완",
    )),
    Module("design", "03. 플랫폼 설계", "LM2001070103", ("requirements",), (
        "인프라 설계 준비", "인프라 설계", "기능 정의", "기능 설계", "인터페이스 설계 준비", "인터페이스 설계",
        "지식화 구현 설계 준비", "지식화 구현 설계",
    )),
    Module("infrastructure", "04. 인프라 구현", "LM2001070104", ("design",), (
        "하드웨어 환경 사전 계획", "하드웨어 환경 구현", "소프트웨어 환경 사전 계획", "소프트웨어 환경 구현",
        "네트워크 환경 사전 계획", "네트워크 환경 구현",
    )),
    Module("function", "05. 기능 구현", "LM2001070105", ("design",), (
        "인공지능 학습 기능 구현", "인공지능 추론 기능 구현", "인공지능 인지 기능 구현",
        "인공지능 모델링 기능 구현", "인공지능 플랫폼 관리 기능 구현",
    )),
    Module("interface", "06. 인터페이스 구현", "LM2001070106", ("design",), (
        "휴먼머신 인터랙션 분석 및 환경 구성", "휴먼머신 인터랙션 구현 및 테스트",
        "내부 인터페이스 분석 및 환경 구성", "내부 인터페이스 구현 및 테스트",
        "외부 인터페이스 분석 및 환경 구성", "외부 인터페이스 구현 및 테스트",
    )),
    Module("knowledge", "07. 지식화 구현", "LM2001070107", ("design",), (
        "데이터 수집 기능 구현", "데이터 유형별 수집 모듈 구현", "로깅·검증·예외 처리 모듈 구현",
        "데이터 추출 모듈 구현", "데이터 변환 모듈 구현", "학습 데이터 적재 모듈 구현", "데이터 검증 모듈 구현",
        "학습 데이터 저장 모듈 구현", "저장소 생성·변경·삭제 기능 모듈 구현", "학습 데이터 관리 기능 구현",
        "지식 표현", "지식 베이스 구축 모듈 구현", "지식 베이스 생성", "지식 베이스 저장·변경·검색·시각화 모듈 구현", "지식 베이스 관리 기능 모듈 구현",
    )),
    Module("test", "08. 플랫폼 테스트", "LM2001070108", ("infrastructure", "function", "interface", "knowledge"), (
        "단위 테스트 계획", "단위 테스트 수행", "통합 테스트 계획", "통합 테스트 수행", "인공지능 플랫폼 성능 테스트", "인수 테스트",
    )),
    Module("quality", "09. 품질 관리", "LM2001070109", ("test",), (
        "품질 관리 계획서 작성", "점검표 작성", "품질 보증 활동 수행", "품질 평가", "품질 개선 활동 수행", "프로세스 통제",
    )),
)


def all_units() -> list[LearningUnit]:
    return [LearningUnit(module.module_id, f"{module.module_id}_{index}", name)
            for module in MODULES for index, name in enumerate(module.units, start=1)]


def module_for_source(source: str) -> Module | None:
    return next((module for module in MODULES if module.filename_prefix in source), None)


def module_name(module_id: str) -> str:
    return next((module.name for module in MODULES if module.module_id == module_id), module_id)
