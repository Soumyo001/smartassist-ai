import sqlite3

from pathlib import Path
from datetime import datetime, timezone

from .config import APP_DB


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")


def initialize_database(
    path: Path = APP_DB
) -> None:

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(path)

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'personal',
                action_type TEXT NOT NULL DEFAULT 'human_required',
                status TEXT NOT NULL DEFAULT 'pending',
                assignee TEXT NOT NULL DEFAULT 'self',
                priority TEXT NOT NULL DEFAULT 'normal',
                due_at_utc TEXT,
                due_date_local TEXT,
                timezone TEXT NOT NULL DEFAULT 'Asia/Dhaka',
                time_precision TEXT NOT NULL DEFAULT 'unscheduled',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        conn.commit()

    finally:
        conn.close()


def seed_once(path: Path = APP_DB) -> int:
    """Create a sample task without duplicates."""

    initialize_database(path)

    conn = sqlite3.connect(path)

    try:
        now = utc_now()

        conn.execute(
            """
            INSERT OR IGNORE INTO tasks
            (
                id,
                title,
                description,
                status,
                created_at_utc,
                updated_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "step1-persist-test",
                "Persistent memory smoke test",
                "This task must survive restarting Python.",
                "pending",
                now,
                now
            )
        )

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0]

        return int(count)

    finally:
        conn.close()