
import json
from datetime import date

from task_agent.agent import ActionPlan, run_turn
from task_agent.storage import create_task, list_tasks

from task_agent.response_formatting import format_task_list

from task_agent.task_intelligence import (
    audit_last_priority_list,
    standalone_due_date,
)


def example_list(priorities):
    tasks = [
        {
            "id": index,
            "title": f"Task {index}",
            "status": "To do",
            "priority": priority,
            "due_date": None,
        }
        for index, priority in enumerate(
            priorities, 1
        )
    ]

    return (
        "*Sorted by priority: Low → High → Medium*\n\n"
        + format_task_list(tasks)
    )


def test_audit_ignores_later_conversational_summary():

    history = [
        example_list(["Low", "High", "Medium"]),
        (
            "The last list was sorted by priority: "
            "Low first, then High, then Medium."
        ),
    ]

    result = audit_last_priority_list(history)

    assert "**Yes.**" in result


def test_audit_detects_actual_wrong_order():

    history = [
        example_list(["High", "Low", "Medium"]),
        "The last list was sorted by priority.",
    ]

    result = audit_last_priority_list(history)

    assert "**No.**" in result


def test_standalone_date_parser_is_narrow():

    today = date(2026, 10, 27)

    assert standalone_due_date(
        "16th november", today
    ) == "2026-11-16"

    assert standalone_due_date(
        "31st november", today
    ) is None

    assert standalone_due_date(
        "16th October", today
    ) is None

    assert standalone_due_date(
        "16th November and delete the task",
        today,
    ) is None


def test_date_reply_updates_only_previous_task(tmp_path):

    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "update_task",
        ]
    }))

    other = create_task(
        title="Other task",
        path=db,
    )

    target = create_task(
        title="Submit assignment",
        path=db,
    )

    class FakePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return ActionPlan(
                mode="conversation",
                message=(
                    "NEVER isn't a valid date format. "
                    "Please provide a specific due date."
                ),
            )

    planner = FakePlanner()

    args = {
        "planner": planner,
        "thread_id": "date-followup",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    run_turn(
        f"Update task #{target['id']} due date to NEVER",
        **args,
    )

    result = run_turn(
        "16th November",
        **args,
    )

    tasks = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert tasks[target["id"]]["due_date"] == "2026-11-16"
    assert tasks[other["id"]]["due_date"] is None

    assert "Task updated" in result
    assert planner.calls == 1
