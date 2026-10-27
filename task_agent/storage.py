import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "smartassist.sqlite3"


def init_db(path: Path = DB_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                title TEXT NOT NULL
                    CHECK(length(trim(title)) > 0),

                description TEXT NOT NULL DEFAULT '',

                status TEXT NOT NULL DEFAULT 'To do'
                    CHECK(status IN (
                        'To do', 'In progress', 'Done'
                    )),

                priority TEXT NOT NULL DEFAULT 'Medium'
                    CHECK(priority IN (
                        'Low', 'Medium', 'High'
                    )),

                due_date TEXT,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                updated_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()


def list_tasks(path: Path = DB_PATH) -> list[dict]:
    init_db(path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT * FROM tasks
            ORDER BY
                CASE status
                    WHEN 'To do' THEN 0
                    WHEN 'In progress' THEN 1
                    ELSE 2
                END,
                id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
