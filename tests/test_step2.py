import json
import pytest

from pydantic import ValidationError

from task_agent.storage import (
    create_task,
    get_task,
    list_tasks,
    update_task,
    complete_task,
    delete_task,
)

from task_agent.tools import build_allowed_tools


def test_crud_lifecycle_and_validation(tmp_path):
    db = tmp_path / "tasks.sqlite3"

    created = create_task(
        title="  Finish presentation  ",
        priority="High",
        due_date="2026-10-28",
        path=db,
    )

    task_id = created["id"]

    assert created["title"] == "Finish presentation"
    assert get_task(task_id, db)["priority"] == "High"
    assert len(list_tasks(db)) == 1

    updated = update_task(
        task_id,
        status="In progress",
        priority="Low",
        path=db,
    )

    assert updated["status"] == "In progress"
    assert updated["priority"] == "Low"

    assert complete_task(
        task_id, db
    )["status"] == "Done"

    assert len(
        list_tasks(db, status="Done")
    ) == 1

    assert update_task(
        task_id,
        due_date=None,
        path=db
    )["due_date"] is None

    assert delete_task(
        task_id, db
    )["id"] == task_id

    assert get_task(task_id, db) is None

    with pytest.raises(KeyError):
        delete_task(task_id, db)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": "   "},
        {"title": "Test", "priority": "Urgent"},
        {"title": "Test", "due_date": "tomorrow"},
        {"title": "Test", "due_date": "2026-02-30"},
    ]
)
def test_invalid_create_is_blocked(
    tmp_path, kwargs
):

    with pytest.raises(ValidationError):
        create_task(
            path=tmp_path / "bad.sqlite3",
            **kwargs
        )


def test_tool_allowlist_enforced(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    policy = tmp_path / "permissions.json"

    policy.write_text(
        json.dumps({
            "enabled_tools": [
                "create_task",
                "list_tasks"
            ]
        })
    )

    tools = build_allowed_tools(db, policy)

    assert {
        t.name for t in tools
    } == {
        "create_task",
        "list_tasks"
    }

    create_tool = next(
        t for t in tools
        if t.name == "create_task"
    )

    result = create_tool.invoke({
        "title": "Allowed task"
    })

    assert json.loads(result)["title"] == "Allowed task"

    # Revoke permission after creating the tool object.
    policy.write_text(
        json.dumps({
            "enabled_tools": ["list_tasks"]
        })
    )

    with pytest.raises(PermissionError):
        create_tool.invoke({
            "title": "Blocked task"
        })

    assert len(list_tasks(db)) == 1