"""Label indexed NCS passages with their curriculum unit from document structure.

NCS learning modules number each competency element ("1-1. 요구사항 수집하기")
and repeat it on its practice pages ("수행 내용 / 요구사항 수집하기").  A
passage belongs to the most recent heading before it in the same module, so
these labels come from the official documents, not from any model.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .curriculum import MODULES, LearningUnit, all_units

HEADING = re.compile(r"(?:^|\n)\s*\d+-\d+\.\s*\n?\s*([^\n]{2,40})|수행 내용\s*/\s*([^\n]{2,40})")
_SUFFIX = re.compile(r"(을\s*)?하기$|기$")


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return {compact[index:index + 2] for index in range(len(compact) - 1)}


def _dice(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    return 2 * len(a & b) / max(1, len(a) + len(b))


def match_unit(heading: str, units: list[LearningUnit], threshold: float = 0.6) -> LearningUnit | None:
    name = _SUFFIX.sub("", heading.strip())
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", name)
    # A heading may prefix the unit name, e.g. "인공지능 플랫폼 인수 테스트".
    contained = [unit for unit in units if compact.endswith(re.sub(r"[^0-9A-Za-z가-힣]", "", unit.name))]
    if contained:
        return max(contained, key=lambda unit: len(unit.name))
    best = max(units, key=lambda unit: _dice(name, unit.name), default=None)
    return best if best is not None and _dice(name, best.name) >= threshold else None


def label_chunks(chunks: list[dict]) -> dict[str, str]:
    """Map ``chunk_id`` to ``unit_id`` for passages after a recognised heading."""
    labels: dict[str, str] = {}
    by_module = {module.module_id: [unit for unit in all_units() if unit.module_id == module.module_id]
                 for module in MODULES}
    current: dict[str, str | None] = {}
    for chunk in chunks:  # chunks are stored in document order
        module = next((module for module in MODULES if module.filename_prefix in chunk["source"]), None)
        if module is None or chunk["text"].count("·") >= 20:
            continue
        before = current.get(module.module_id)
        switch_at = None
        for match in HEADING.finditer(chunk["text"]):
            unit = match_unit(match.group(1) or match.group(2), by_module[module.module_id])
            if unit is not None:
                current[module.module_id] = unit.unit_id
                switch_at = switch_at if switch_at is not None else match.start()
        # A heading in the back half starts the next unit; most of this
        # passage still belongs to the previous one.
        late = switch_at is not None and switch_at > len(chunk["text"]) / 2 and before
        label = before if late else current.get(module.module_id)
        if label:
            labels[chunk["chunk_id"]] = label
    return labels


if __name__ == "__main__":
    chunks = json.loads(Path("data/index/chunks.json").read_text(encoding="utf-8"))
    labels = label_chunks(chunks)
    counts = Counter(labels.values())
    units = all_units()
    print(f"labeled {len(labels)} / {len(chunks)} passages, units covered {len(counts)} / {len(units)}")
    for unit in units:
        print(f"  {counts.get(unit.unit_id, 0):3d}  {unit.name}")
