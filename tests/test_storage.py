import sqlite3
import pytest

from task_agent.storage import init_db, list_tasks


def test_task_persists(tmp_path):

    path = tmp_path / "tasks.sqlite3"
    init_db(path)

    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO tasks "
            "(title, priority) VALUES (?, ?)",
            ("Write README", "High")
        )
        conn.commit()

    rows = list_tasks(path)

    assert len(rows) == 1
    assert rows[0]["title"] == "Write README"
    assert rows[0]["id"] == 1


def test_invalid_priority_rejected(tmp_path):

    path = tmp_path / "tasks.sqlite3"
    init_db(path)

    with sqlite3.connect(path) as conn:

        with pytest.raises(
            sqlite3.IntegrityError
        ):
            conn.execute(
                "INSERT INTO tasks "
                "(title, priority) VALUES (?, ?)",
                ("Bad", "Urgent")
            )