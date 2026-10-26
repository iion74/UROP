from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProgressSummary:
    total_units: int
    completed_units: int
    review_units: int
    total_minutes: int

    @property
    def coverage(self) -> float:
        return 100 * self.completed_units / self.total_units if self.total_units else 0.0


class ProgressStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS study_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                module_id TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                unit_name TEXT NOT NULL,
                status TEXT NOT NULL,
                minutes INTEGER NOT NULL,
                note TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def save(self, module_id: str, unit_id: str, unit_name: str, status: str, minutes: int, note: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO study_log(created_at,module_id,unit_id,unit_name,status,minutes,note) VALUES(?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), module_id, unit_id, unit_name, status, minutes, note),
            )

    def latest_statuses(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT unit_id, status FROM study_log
                WHERE id IN (SELECT MAX(id) FROM study_log GROUP BY unit_id)""").fetchall()
        return dict(rows)

    def summary(self, total_units: int) -> ProgressSummary:
        statuses = self.latest_statuses()
        with self._connect() as connection:
            minutes = connection.execute("SELECT COALESCE(SUM(minutes), 0) FROM study_log").fetchone()[0]
        return ProgressSummary(total_units, sum(value == "완료" for value in statuses.values()),
                               sum(value == "복습 필요" for value in statuses.values()), minutes)

    def recent_logs(self, limit: int = 8) -> list[tuple]:
        with self._connect() as connection:
            return connection.execute("""SELECT created_at, unit_name, status, minutes, note FROM study_log
                ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
