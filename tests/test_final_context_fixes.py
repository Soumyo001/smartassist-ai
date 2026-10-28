
import json
from datetime import date

from task_agent.agent import ActionPlan, Operation, run_turn
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

def test_date_reply_after_backend_validation_error(tmp_path):
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
        title="Unrelated task",
        path=db,
    )

    target = create_task(
        title="Submit assignment",
        path=db,
    )

    class InvalidDatePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                operations=[
                    Operation(
                        action="update_task",
                        task_id=target["id"],
                        due_date="NEVER",
                    )
                ]
            )

    planner = InvalidDatePlanner()

    args = {
        "planner": planner,
        "thread_id": "backend-date-rejection",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        f"Update task #{target['id']} due date to NEVER",
        **args,
    )

    assert "Could not update_task" in first

    tasks_before = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert tasks_before[target["id"]]["due_date"] is None
    assert tasks_before[other["id"]]["due_date"] is None

    second = run_turn(
        "16th November",
        **args,
    )

    tasks_after = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert "Task updated" in second

    assert (
        tasks_after[target["id"]]["due_date"]
        == "2026-11-16"
    )

    assert tasks_after[other["id"]]["due_date"] is None

    # The valid follow-up should be handled
    # deterministically, without another LLM call.
    assert planner.calls == 1


def test_never_date_cannot_be_silently_cleared(tmp_path):
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
        title="QA Control",
        due_date="2026-11-20",
        path=db,
    )

    target = create_task(
        title="QA Date Target",
        due_date="2026-11-10",
        path=db,
    )

    class ClearDatePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return ActionPlan(
                operations=[
                    Operation(
                        action="update_task",
                        task_id=target["id"],
                        clear_due_date=True,
                    )
                ]
            )

    planner = ClearDatePlanner()

    args = {
        "planner": planner,
        "thread_id": "never-clear-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    # Simulate the problematic model interpretation:
    # NEVER becomes clear_due_date=True.
    first = run_turn(
        f"Update task #{target['id']} due date to NEVER",
        **args,
    )

    assert "Could not update_task" in first
    assert "Invalid due date" in first
    assert "Task updated" not in first

    tasks = {
        task["id"]: task
        for task in list_tasks(db)
    }

    # Neither task may be changed by the invalid request.
    assert tasks[target["id"]]["due_date"] == "2026-11-10"
    assert tasks[other["id"]]["due_date"] == "2026-11-20"

    # Immediately provide a valid standalone date.
    second = run_turn(
        "16th November",
        **args,
    )

    tasks = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert "Task updated" in second
    assert tasks[target["id"]]["due_date"] == "2026-11-16"
    assert tasks[other["id"]]["due_date"] == "2026-11-20"

    # Follow-up must not require another planner call.
    assert planner.calls == 1

    # Genuine date clearing must continue to work.
    third = run_turn(
        f"Clear the due date for task #{other['id']}",
        **args,
    )

    tasks = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert "Task updated" in third
    assert tasks[other["id"]]["due_date"] is None
    assert tasks[target["id"]]["due_date"] == "2026-11-16"
    assert planner.calls == 2


def test_date_followup_after_groq_double_failure(tmp_path):
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

    target = create_task(
        title="QA Date Target",
        due_date="2026-11-10",
        path=db,
    )

    control = create_task(
        title="QA Date Control",
        due_date="2026-11-20",
        path=db,
    )

    class BadRequestError(Exception):
        body = {
            "error": {
                "code": "json_validate_failed"
            }
        }

    class FailingPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise BadRequestError("json_validate_failed")

    planner = FailingPlanner()

    args = {
        "planner": planner,
        "thread_id": "groq-date-double-failure",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        f"Update task #{target['id']} due date to NEVER",
        **args,
    )

    assert planner.calls == 2
    assert "Invalid due date" in first
    assert "No task changes were made" in first

    before = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert before[target["id"]]["due_date"] == "2026-11-10"
    assert before[control["id"]]["due_date"] == "2026-11-20"

    second = run_turn(
        "16th November",
        **args,
    )

    after = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert "Task updated" in second
    assert after[target["id"]]["due_date"] == "2026-11-16"
    assert after[control["id"]]["due_date"] == "2026-11-20"

    # No additional Groq invocation for the follow-up.
    assert planner.calls == 2
