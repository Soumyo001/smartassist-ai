import sqlite3
from pathlib import Path

from .schemas import Status, TaskCreate, TaskUpdate

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "smartassist.sqlite3"


def init_db(path: Path = DB_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute("""
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


def _require_id(task_id: int) -> None:
    if type(task_id) is not int or task_id < 1:
        raise ValueError(
            "task_id must be a positive integer"
        )


def get_task(
    task_id: int,
    path: Path = DB_PATH
) -> dict | None:

    _require_id(task_id)
    init_db(path)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,)
        ).fetchone()

        return dict(row) if row else None


def list_tasks(
    path: Path = DB_PATH,
    status: Status | None = None
) -> list[dict]:

    init_db(path)

    if status is not None and status not in (
        "To do", "In progress", "Done"
    ):
        raise ValueError("Invalid task status")

    sql = "SELECT * FROM tasks"
    params = []

    if status is not None:
        sql += " WHERE status = ?"
        params.append(status)

    sql += """
        ORDER BY
            CASE status
                WHEN 'To do' THEN 0
                WHEN 'In progress' THEN 1
                ELSE 2
            END,
            id DESC
    """

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            sql, params
        ).fetchall()

        return [dict(row) for row in rows]


def create_task(
    *,
    title: str,
    description: str = "",
    status: Status = "To do",
    priority: str = "Medium",
    due_date: str | None = None,
    path: Path = DB_PATH
) -> dict:

    data = TaskCreate(
        title=title,
        description=description,
        status=status,
        priority=priority,
        due_date=due_date
    ).model_dump(mode="json")

    init_db(path)

    with sqlite3.connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO tasks
                (title, description, status, priority, due_date)
            VALUES
                (:title, :description, :status,
                 :priority, :due_date)
            """,
            data
        )

        task_id = cursor.lastrowid

    return get_task(task_id, path)


def update_task(
    task_id: int,
    *,
    path: Path = DB_PATH,
    **changes
) -> dict:

    _require_id(task_id)

    if not changes:
        raise ValueError(
            "At least one field must be supplied"
        )

    fields = TaskUpdate.model_validate(
        changes
    ).model_dump(
        mode="json",
        exclude_unset=True
    )

    if not fields:
        raise ValueError("No fields to update")

    # Column names are restricted by the Pydantic schema.
    assignments = ", ".join(
        f"{column} = ?" for column in fields
    )

    init_db(path)

    with sqlite3.connect(path) as conn:
        cursor = conn.execute(
            f"""
            UPDATE tasks
            SET {assignments},
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            [*fields.values(), task_id]
        )

        if cursor.rowcount != 1:
            raise KeyError(
                f"Task #{task_id} was not found"
            )

    return get_task(task_id, path)


def complete_task(
    task_id: int,
    path: Path = DB_PATH
) -> dict:

    return update_task(
        task_id,
        path=path,
        status="Done"
    )


def delete_task(
    task_id: int,
    path: Path = DB_PATH
) -> dict:

    _require_id(task_id)

    task = get_task(task_id, path)

    if task is None:
        raise KeyError(
            f"Task #{task_id} was not found"
        )

    with sqlite3.connect(path) as conn:
        conn.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task_id,)
        )

    return task