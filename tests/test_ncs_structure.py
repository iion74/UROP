from src.curriculum import all_units
from src.ncs_structure import label_chunks, match_unit

SOURCE = "ncs/LM2001070102_요구사항+분석.pdf"
REQUIREMENTS = [unit for unit in all_units() if unit.module_id == "requirements"]


def _chunk(number: int, text: str, source: str = SOURCE) -> dict:
    return {"chunk_id": f"{source}:{number}", "source": source, "text": text}


def test_heading_variants_match_curriculum_units():
    assert match_unit("요구사항 수집하기", REQUIREMENTS).unit_id == "requirements_1"
    tests = [unit for unit in all_units() if unit.module_id == "test"]
    assert match_unit("인공지능 플랫폼 인수 테스트하기", tests).unit_id == "test_6"
    assert match_unit("전혀 다른 제목", REQUIREMENTS) is None


def test_passages_follow_the_most_recent_section_heading():
    chunks = [
        _chunk(0, "목차 " + "·" * 30 + " 1-1. 요구사항 수집하기"),
        _chunk(1, "1-1.\n요구사항 수집하기\n요구사항은 이해관계자에게서 모은다." + " 내용" * 40),
        _chunk(2, "수집 기법은 인터뷰와 워크숍이다." + " 내용" * 40),
        # A heading late in a passage starts the next unit with the next passage.
        _chunk(3, "인터뷰 결과를 정리한다." + " 내용" * 80 + "\n1-2.\n우선순위 도출하기"),
        _chunk(4, "MoSCoW 기법으로 분류한다."),
        _chunk(5, "수행 내용 / 요구사항 정의하기\n기능 요구사항을 정의한다."),
        _chunk(6, "다른 모듈 내용", source="ncs/LM2001070108_테스트.pdf"),
    ]
    labels = label_chunks(chunks)
    assert f"{SOURCE}:0" not in labels  # contents pages are skipped
    assert [labels[f"{SOURCE}:{number}"] for number in (1, 2, 3, 4, 5)] == [
        "requirements_1", "requirements_1", "requirements_1", "requirements_2", "requirements_3",
    ]
    assert "ncs/LM2001070108_테스트.pdf:6" not in labels  # no heading seen in that module yet
