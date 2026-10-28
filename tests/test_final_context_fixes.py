
import json
from datetime import date

from task_agent.agent import ActionPlan, Operation, run_turn
from task_agent.storage import create_task, list_tasks, get_task

from task_agent.response_formatting import format_task_list

from task_agent.task_intelligence import (
    audit_last_priority_list,
    standalone_due_date,
    mentioned_task_ids,
    is_bulk_delete_request,
    is_explicit_delete_request,
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


def test_date_followup_after_conversational_clarification(tmp_path):
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

    class ClarificationPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                mode="conversation",
                message=(
                    "I'm not sure what you mean by setting "
                    'the due date to "NEVER". '
                    "Do you want to remove the due date "
                    "entirely, or set it to a specific date?"
                ),
            )

    planner = ClarificationPlanner()

    args = {
        "planner": planner,
        "thread_id": "date-clarification-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        f"Update task #{target['id']} due date to NEVER",
        **args,
    )

    assert "specific date" in first

    second = run_turn(
        "18th November",
        **args,
    )

    tasks = {
        task["id"]: task
        for task in list_tasks(db)
    }

    assert "Task updated" in second
    assert tasks[target["id"]]["due_date"] == "2026-11-18"
    assert tasks[control["id"]]["due_date"] == "2026-11-20"

    assert planner.calls == 1


def test_multiple_deletions_are_blocked_once(tmp_path):
    from task_agent.task_intelligence import mentioned_task_ids

    assert mentioned_task_ids(
        "delete task #17 and #18"
    ) == {17, 18}

    assert mentioned_task_ids(
        "remove task ID 1,2,3"
    ) == {1, 2, 3}

    assert mentioned_task_ids(
        "set ID 15's due date 2026-11-10"
    ) == {15}

    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    first = create_task(
        title="Alpha",
        path=db,
    )

    second = create_task(
        title="Beta",
        path=db,
    )

    class MultiDeletePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                operations=[
                    Operation(
                        action="delete_task",
                        target="Alpha",
                    ),
                    Operation(
                        action="delete_task",
                        target="Beta",
                    ),
                ]
            )

    planner = MultiDeletePlanner()

    args = {
        "planner": planner,
        "thread_id": "multi-delete-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    # Explicitly named IDs: block before planning.
    reply = run_turn(
        f"delete task #{first['id']} "
        f"and #{second['id']}",
        **args,
    )

    assert reply.count("For safety") == 1
    assert "No tasks were deleted" in reply
    assert planner.calls == 0

    # Comma-separated ID list: also block.
    reply = run_turn(
        f"remove task ID {first['id']},{second['id']}",
        **args,
    )

    assert reply.count("For safety") == 1
    assert planner.calls == 0

    # Two named tasks: block before planning.
    reply = run_turn(
        "remove task Alpha and task Beta",
        **args,
    )

    assert reply.count("For safety") == 1
    assert planner.calls == 0

    # An ambiguous model-generated two-delete plan
    # must also be rejected as a single response.
    reply = run_turn(
        "remove Alpha",
        **args,
    )

    assert reply.count("For safety") == 1
    assert planner.calls == 1

    remaining = {
        task["id"]
        for task in list_tasks(db)
    }

    assert remaining == {
        first["id"],
        second["id"],
    }


def test_bulk_delete_phrasings_are_blocked_before_planning(
    tmp_path,
):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    first = create_task(title="Alpha", path=db)
    second = create_task(title="Beta", path=db)

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Bulk deletion must not invoke the planner"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "bulk-language-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    requests = [
        "delete all my tasks",
        "delete task A and B",
        "remove all completed and pending tasks",
        "delete both tasks",
        "remove Alpha and Beta",
        "delete task #17 and #18",
        "remove task ID 17,18",
        'delete "Alpha" "Beta"',
    ]

    for message in requests:
        result = run_turn(message, **args)

        assert result.count("For safety") == 1
        assert "No tasks were deleted" in result

    remaining = {
        task["id"]
        for task in list_tasks(db)
    }

    assert remaining == {
        first["id"],
        second["id"],
    }

    assert not is_bulk_delete_request(
        'delete task "Research and Development"'
    )

    assert not is_explicit_delete_request(
        "Don't delete task #17"
    )


def test_ambiguous_delete_resumes_by_option(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    first = create_task(
        title="QA Archive Alpha",
        path=db,
    )

    second = create_task(
        title="QA Archive Beta",
        path=db,
    )

    class DeletePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                operations=[
                    Operation(
                        action="delete_task",
                        target="QA Archive",
                    )
                ]
            )

    planner = DeletePlanner()

    args = {
        "planner": planner,
        "thread_id": "ambiguous-delete-option",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    initial = run_turn(
        "Delete the QA Archive task",
        **args,
    )

    assert "Which task did you mean?" in initial

    # Clarification must not modify any task.
    assert get_task(first["id"], db) is not None
    assert get_task(second["id"], db) is not None

    # list_tasks orders these To do tasks by descending ID.
    # Therefore option 1 refers to the second-created task.
    selected = run_turn("1", **args)

    assert "Task deleted" in selected
    assert get_task(second["id"], db) is None
    assert get_task(first["id"], db) is not None

    # Selecting a displayed option must not call Groq.
    assert planner.calls == 1


def test_ambiguous_delete_cancel_and_explicit_id(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    first = create_task(
        title="QA Archive Alpha",
        path=db,
    )

    second = create_task(
        title="QA Archive Beta",
        path=db,
    )

    class DeletePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return ActionPlan(
                operations=[
                    Operation(
                        action="delete_task",
                        target="QA Archive",
                    )
                ]
            )

    planner = DeletePlanner()

    args = {
        "planner": planner,
        "thread_id": "ambiguous-delete-id",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first_reply = run_turn(
        "Delete the QA Archive task",
        **args,
    )

    assert "Which task did you mean?" in first_reply

    # An ID outside the candidate list is invalid.
    invalid = run_turn("#999", **args)

    assert "Which task did you mean?" in invalid
    assert len(list_tasks(db)) == 2

    # Cancelling must discard the pending deletion.
    cancelled = run_turn("cancel that", **args)

    assert "Cancelled" in cancelled
    assert len(list_tasks(db)) == 2

    # Start a fresh explicit deletion request.
    clarification = run_turn(
        "Delete the QA Archive task",
        **args,
    )

    assert "Which task did you mean?" in clarification

    # Select an actual database ID rather than option 1.
    result = run_turn(
        f"#{first['id']}",
        **args,
    )

    assert "Task deleted" in result
    assert get_task(first["id"], db) is None
    assert get_task(second["id"], db) is not None

    assert planner.calls == 2


def test_negated_delete_and_quoted_single_title(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    task = create_task(
        title="Research and Development",
        path=db,
    )

    class DeletePlanner:
        def invoke(self, messages):
            return ActionPlan(
                operations=[
                    Operation(
                        action="delete_task",
                        target="Research and Development",
                    )
                ]
            )

    args = {
        "planner": DeletePlanner(),
        "thread_id": "quoted-delete-safety",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    # Even a faulty LLM deletion plan must not override
    # an explicitly negated user instruction.
    blocked = run_turn(
        "Don't delete Research and Development",
        **args,
    )

    assert "Could not delete_task" in blocked
    assert get_task(task["id"], db) is not None

    # A single quoted title containing "and" is valid.
    removed = run_turn(
        'Delete task "Research and Development"',
        **args,
    )

    assert "Task deleted" in removed
    assert get_task(task["id"], db) is None


def test_groq_conversation_delete_clarification_option(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    alpha = create_task(
        title="QA Deletion Probe Alpha",
        path=db,
    )

    beta = create_task(
        title="QA Deletion Probe Beta",
        path=db,
    )

    class ConversationPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                mode="conversation",
                message=(
                    "Which QA Deletion Probe task would "
                    "you like to delete? Alpha or Beta?"
                ),
            )

    planner = ConversationPlanner()

    args = {
        "planner": planner,
        "thread_id": "conversation-delete-option",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        "Delete the QA Deletion Probe task",
        **args,
    )

    assert "Which task did you mean?" in first
    assert f"#{alpha['id']}" in first
    assert f"#{beta['id']}" in first

    # Clarification must not delete either task.
    assert get_task(alpha["id"], db) is not None
    assert get_task(beta["id"], db) is not None

    # The backend lists these To do tasks by
    # descending ID, so option 1 is Beta.
    second = run_turn("1", **args)

    assert "Task deleted" in second
    assert f"#{beta['id']}" in second

    assert get_task(beta["id"], db) is None
    assert get_task(alpha["id"], db) is not None

    # No extra Groq call for the selection.
    assert planner.calls == 1


def test_groq_conversation_delete_cancel_and_id(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    alpha = create_task(
        title="QA Deletion Probe Alpha",
        path=db,
    )

    beta = create_task(
        title="QA Deletion Probe Beta",
        path=db,
    )

    class ConversationPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(
                mode="conversation",
                message=(
                    "Which task would you like "
                    "to delete? Alpha or Beta?"
                ),
            )

    planner = ConversationPlanner()

    args = {
        "planner": planner,
        "thread_id": "conversation-delete-cancel",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        "Delete the QA Deletion Probe task",
        **args,
    )

    assert "Which task did you mean?" in first

    # An ID not displayed as a candidate must
    # never authorize deletion.
    invalid = run_turn("#999", **args)

    assert "Which task did you mean?" in invalid
    assert len(list_tasks(db)) == 2

    # Cancellation must discard authorization.
    cancelled = run_turn("cancel that", **args)

    assert "Cancelled" in cancelled
    assert len(list_tasks(db)) == 2

    # Start a completely new request.
    second_request = run_turn(
        "Delete the QA Deletion Probe task",
        **args,
    )

    assert "Which task did you mean?" in second_request

    # Now choose an actual displayed task ID.
    selected = run_turn(
        f"#{alpha['id']}",
        **args,
    )

    assert "Task deleted" in selected

    assert get_task(alpha["id"], db) is None
    assert get_task(beta["id"], db) is not None

    assert planner.calls == 2


def test_groq_cannot_autoselect_ambiguous_delete(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    alpha = create_task(
        title="QA Deletion Probe Alpha",
        path=db,
    )

    beta = create_task(
        title="QA Deletion Probe Beta",
        path=db,
    )

    class GuessingPlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            # Simulate the dangerous real behavior:
            # Groq chooses Alpha without user selection.
            return ActionPlan(
                mode="actions",
                operations=[
                    Operation(
                        action="delete_task",
                        target="QA Deletion Probe Alpha",
                    )
                ]
            )

    planner = GuessingPlanner()

    args = {
        "planner": planner,
        "thread_id": "unsafe-delete-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    first = run_turn(
        "Delete the QA Deletion Probe task",
        **args,
    )

    # Must clarify even though Groq chose Alpha.
    assert "Which task did you mean?" in first
    assert f"#{alpha['id']}" in first
    assert f"#{beta['id']}" in first
    assert "Task deleted" not in first

    # Absolutely no deletion before selection.
    assert get_task(alpha["id"], db) is not None
    assert get_task(beta["id"], db) is not None

    # Option 1 corresponds to Beta (descending ID).
    second = run_turn("1", **args)

    assert "Task deleted" in second
    assert f"#{beta['id']}" in second

    assert get_task(alpha["id"], db) is not None
    assert get_task(beta["id"], db) is None

    # Selection must not require Groq again.
    assert planner.calls == 1


def test_partial_delete_title_cannot_authorize_model_guess(
    tmp_path,
):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
            "delete_task",
        ]
    }))

    beta = create_task(
        title="QA Deletion Probe Beta",
        path=db,
    )

    class GuessingPlanner:
        def invoke(self, messages):
            return ActionPlan(
                mode="actions",
                operations=[
                    Operation(
                        action="delete_task",
                        target="QA Deletion Probe Beta",
                    )
                ]
            )

    args = {
        "planner": GuessingPlanner(),
        "thread_id": "partial-delete-guard",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    result = run_turn(
        "Delete the QA Deletion Probe task",
        **args,
    )

    # A partial name is not permission for Groq
    # to choose the remaining task by itself.
    assert "Could not delete_task" in result
    assert "No task was deleted" in result
    assert "Task deleted" not in result

    assert get_task(beta["id"], db) is not None


def test_empty_priority_list_is_correctly_audited():
    # Empty task lists must include their count.
    empty_list = format_task_list([])

    assert empty_list == (
        "**Your tasks (0)**\n\n"
        "No matching tasks found."
    )

    # Custom headings must remain consistent.
    assert format_task_list(
        [],
        heading="Related tasks",
    ) == (
        "**Related tasks (0)**\n\n"
        "No matching tasks found."
    )

    # Simulate a real priority-sorted response.
    history = [
        (
            "*Sorted by priority: "
            "Low → High → Medium*\n\n"
            + empty_list
        ),
        "The previous list contained no matching tasks.",
    ]

    result = audit_last_priority_list(history)

    assert result == (
        "The previous task list was empty, "
        "so no ordering was needed."
    )

    # Existing nonempty formatting must still work.
    nonempty_list = example_list(
        ["Low", "High", "Medium"]
    )

    assert "**Your tasks (3)**" in nonempty_list
    assert "**Yes.**" in audit_last_priority_list(
        [nonempty_list]
    )


def test_status_filtered_lists_and_empty_priority_audit(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["list_tasks"]
    }))

    todo = create_task(
        title="QA Todo Row",
        status="To do",
        priority="Low",
        path=db,
    )

    done = create_task(
        title="QA Completed Row",
        status="Done",
        priority="High",
        path=db,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Simple status-filtered lists must not call Groq"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "status-filter-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    # No task is In progress yet.
    empty = run_turn(
        "Show my in-progress tasks sorted by priority: "
        "Low → High → Medium.",
        **args,
    )

    assert "Sorted by priority: Low → High → Medium" in empty
    assert "**Your tasks (0)**" in empty
    assert "No matching tasks found." in empty
    assert "QA Todo Row" not in empty
    assert "QA Completed Row" not in empty

    # Verify that the previous empty-list audit works
    # through the actual LangGraph conversation.
    audit = run_turn(
        "Was your earlier task list correctly sorted by priority?",
        **args,
    )

    assert "previous task list was empty" in audit
    assert "no ordering was needed" in audit

    # Create a matching task and verify only it appears.
    active = create_task(
        title="QA Active Row",
        status="In progress",
        priority="Medium",
        path=db,
    )

    active_list = run_turn(
        "Show my in-progress tasks sorted by priority: "
        "Low → High → Medium.",
        **args,
    )

    assert "**Your tasks (1)**" in active_list
    assert f"#{active['id']}" in active_list
    assert "QA Active Row" in active_list
    assert "QA Todo Row" not in active_list
    assert "QA Completed Row" not in active_list

    # Other status names must work as well.
    completed_list = run_turn(
        "Show completed tasks",
        **args,
    )

    assert "**Your tasks (1)**" in completed_list
    assert f"#{done['id']}" in completed_list
    assert "QA Todo Row" not in completed_list
    assert "QA Active Row" not in completed_list

    todo_list = run_turn(
        "Show to-do tasks",
        **args,
    )

    assert "**Your tasks (1)**" in todo_list
    assert f"#{todo['id']}" in todo_list
    assert "QA Completed Row" not in todo_list
    assert "QA Active Row" not in todo_list

    # Sorting all tasks by status is not a filter.
    all_tasks = run_turn(
        "Show all tasks sorted by status: "
        "To do, In progress, Done",
        **args,
    )

    assert "**Your tasks (3)**" in all_tasks
    assert "QA Todo Row" in all_tasks
    assert "QA Active Row" in all_tasks
    assert "QA Completed Row" in all_tasks



def test_semantic_search_filters_distant_neighbors(
    tmp_path,
    monkeypatch,
):
    from task_agent import vector_store

    db = tmp_path / "tasks.sqlite3"

    documentation = create_task(
        title="Update API documentation",
        path=db,
    )

    unrelated = create_task(
        title="Create a new task",
        path=db,
    )

    class FakeCollection:
        distances = [0.728, 1.572]

        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "ids": [[
                    str(documentation["id"]),
                    str(unrelated["id"]),
                ]],
                "distances": [self.distances],
            }

    collection = FakeCollection()

    monkeypatch.setattr(
        vector_store,
        "sync_task_index",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        vector_store,
        "get_task_collection",
        lambda chroma_path=None: collection,
    )

    relevant = vector_store.search_tasks(
        "API documentation",
        db_path=db,
        chroma_path=tmp_path / "chroma",
    )

    assert len(relevant) == 1
    assert relevant[0]["id"] == documentation["id"]

    # A query with only weak matches must return
    # zero results, not arbitrary nearest tasks.
    collection.distances = [1.699, 1.759]

    irrelevant = vector_store.search_tasks(
        "grocery shopping",
        db_path=db,
        chroma_path=tmp_path / "chroma",
    )

    assert irrelevant == []


def test_topical_search_routing_and_unfinished_filter(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["search_tasks"]
    }))

    documentation = create_task(
        title="Update API documentation",
        status="Done",
        path=db,
    )

    groceries = create_task(
        title="Buy groceries",
        status="To do",
        path=db,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Explicit topical searches should not "
                "require Groq to choose the search tool"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "topical-search-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    # Topic must be extracted and searched semantically.
    related = run_turn(
        "Find tasks related to API documentation.",
        **args,
    )

    assert "**Related tasks (1)**" in related
    assert f"#{documentation['id']}" in related
    assert "Buy groceries" not in related

    # Semantic relevance and unfinished status
    # must both be respected.
    unfinished = run_turn(
        "Show all unfinished tasks related to documentation.",
        **args,
    )

    assert "**Related tasks (0)**" in unfinished
    assert "No matching tasks found." in unfinished
    assert "Update API documentation" not in unfinished
    assert "Buy groceries" not in unfinished

    # A different topic should find the other task.
    shopping = run_turn(
        "Search for tasks about grocery shopping.",
        **args,
    )

    assert "**Related tasks (1)**" in shopping
    assert f"#{groceries['id']}" in shopping
    assert "Update API documentation" not in shopping

def test_sorted_semantic_search_separates_topic_and_order(
    tmp_path,
    monkeypatch,
):
    from task_agent import vector_store

    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["search_tasks"]
    }))

    low = create_task(
        title="Weekly report draft",
        priority="Low",
        path=db,
    )
    high = create_task(
        title="Weekly report review",
        priority="High",
        path=db,
    )
    medium = create_task(
        title="Weekly report summary",
        priority="Medium",
        path=db,
    )

    class ControlledCollection:
        def __init__(self):
            self.returned_ids = [
                medium["id"],
                high["id"],
                low["id"],
            ]
            self.queries = []

        def count(self):
            return len(self.returned_ids)

        def query(self, **kwargs):
            self.queries.append(
                kwargs["query_texts"][0]
            )
            return {
                "ids": [[
                    str(item)
                    for item in self.returned_ids
                ]],
                "distances": [[
                    0.4
                ] * len(self.returned_ids)],
            }

    collection = ControlledCollection()

    monkeypatch.setattr(
        vector_store,
        "sync_task_index",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        vector_store,
        "get_task_collection",
        lambda chroma_path=None: collection,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Explicit topical search should bypass Groq"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "semantic-sort-query-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    prompt = (
        "Find tasks related to weekly reports "
        "in the order: low -> high -> medium"
    )

    ordered = run_turn(prompt, **args)

    # The vector search must receive only the topic.
    assert collection.queries == ["weekly reports"]

    # The displayed list must declare its priority order.
    assert (
        "*Sorted by priority: Low → High → Medium*"
        in ordered
    )
    assert "**Related tasks (3)**" in ordered

    # Chroma returned a different sequence.
    # The agent must apply the user-requested order.
    assert (
        ordered.index(f"#{low['id']}")
        < ordered.index(f"#{high['id']}")
        < ordered.index(f"#{medium['id']}")
    )

    audited = run_turn(
        "was order explicitly as I said ?",
        **args,
    )

    assert "**Yes.**" in audited

    # A singleton result does not demonstrate
    # any relative ordering.
    collection.returned_ids = [medium["id"]]

    single = run_turn(prompt, **args)
    assert "**Related tasks (1)**" in single

    single_audit = run_turn(
        "was order explicitly as I said ?",
        **args,
    )

    assert "Only one task matched" in single_audit
    assert (
        "no relative priority order to verify"
        in single_audit
    )


def test_priority_then_date_descending_list(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["list_tasks"]
    }))

    high = create_task(
        title="High October",
        status="Done",
        priority="High",
        due_date="2026-10-28",
        path=db,
    )

    low_october = create_task(
        title="Low October",
        status="Done",
        priority="Low",
        due_date="2026-10-28",
        path=db,
    )

    low_november = create_task(
        title="Low November",
        status="Done",
        priority="Low",
        due_date="2026-11-16",
        path=db,
    )

    low_undated = create_task(
        title="Low No Date",
        status="Done",
        priority="Low",
        path=db,
    )

    medium = create_task(
        title="Medium No Date",
        status="Done",
        priority="Medium",
        path=db,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Plain listing should not require Groq"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "priority-date-compound-sort",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    prompts = [
        (
            "show all completed tasks in order: "
            "high -> low -> medium "
            "with dates in descending order"
        ),
        (
            "show all completed tasks in order: "
            "high -> low -> medium "
            "with latest date first"
        ),
    ]

    for prompt in prompts:
        result = run_turn(prompt, **args)

        assert "**Your tasks (5)**" in result
        assert (
            "Sorted by priority: High → Low → Medium"
            in result
        )

        positions = [
            result.index(f"#{task['id']}")
            for task in (
                high,
                low_november,
                low_october,
                low_undated,
                medium,
            )
        ]

        assert positions == sorted(positions)

    # Without a secondary date instruction,
    # retain the original priority sorting behavior.
    ordinary = run_turn(
        "Show completed tasks in order: "
        "high -> low -> medium",
        **args,
    )

    assert (
        ordinary.index(f"#{low_october['id']}")
        < ordinary.index(f"#{low_november['id']}")
    )


def test_priority_then_urgent_date_first(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["list_tasks"]
    }))

    # Deliberately create November BEFORE October.
    # This catches accidental sorting by task ID.
    low_november = create_task(
        title="Low November",
        status="Done",
        priority="Low",
        due_date="2026-11-16",
        path=db,
    )

    low_october = create_task(
        title="Low October",
        status="Done",
        priority="Low",
        due_date="2026-10-28",
        path=db,
    )

    low_undated = create_task(
        title="Low Undated",
        status="Done",
        priority="Low",
        path=db,
    )

    high = create_task(
        title="High October",
        status="Done",
        priority="High",
        due_date="2026-10-28",
        path=db,
    )

    medium = create_task(
        title="Medium Undated",
        status="Done",
        priority="Medium",
        path=db,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Plain listing must not call Groq"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "urgent-date-sort-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    prefix = (
        "show all completed tasks in order: "
        "high -> low -> medium "
    )

    def verify_order(result, expected_tasks):
        assert "**Your tasks (5)**" in result

        positions = [
            result.index(f"#{task['id']}")
            for task in expected_tasks
        ]

        assert positions == sorted(positions)

    # Earlier dates should come first within Low priority.
    for instruction in (
        "with urgent dates first",
        "with earliest date first",
        "with dates in ascending order",
    ):
        result = run_turn(prefix + instruction, **args)

        verify_order(result, [
            high,
            low_october,
            low_november,
            low_undated,
            medium,
        ])

    # Newest-first sorting must remain correct.
    for instruction in (
        "with latest date first",
        "with dates in descending order",
    ):
        result = run_turn(prefix + instruction, **args)

        verify_order(result, [
            high,
            low_november,
            low_october,
            low_undated,
            medium,
        ])

    # Without date instructions, the old priority
    # sorting behavior must remain unchanged.
    ordinary = run_turn(
        prefix.strip(),
        **args,
    )

    assert (
        ordinary.index(f"#{low_november['id']}")
        < ordinary.index(f"#{low_october['id']}")
    )


def test_explicit_multi_id_reads_do_not_list_everything(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "list_tasks",
        ]
    }))

    first = create_task(
        title="QA Read Alpha",
        path=db,
    )

    second = create_task(
        title="QA Read Beta",
        path=db,
    )

    unrelated = create_task(
        title="QA Read Unrelated",
        path=db,
    )

    class NoPlanner:
        def invoke(self, messages):
            raise AssertionError(
                "Explicit ID reads should not call Groq"
            )

    args = {
        "planner": NoPlanner(),
        "thread_id": "explicit-multi-id-read",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    prompts = [
        (
            f"Show task #{first['id']} and "
            f"task #{second['id']} separately.",
            [first, second],
        ),
        (
            f"Get details for task #{first['id']}, "
            f"then get details for task #{second['id']}.",
            [first, second],
        ),
        (
            f"Show task #{second['id']} and "
            f"#{first['id']} separately.",
            [second, first],
        ),
    ]

    for prompt, expected in prompts:
        reply = run_turn(prompt, **args)

        # Exactly two task-detail operations.
        assert reply.count("✅ **Task details**") == 2
        assert reply.count("*Action: `get_task`*") == 2

        # Both requested tasks must be present.
        assert expected[0]["title"] in reply
        assert expected[1]["title"] in reply

        # Preserve the order supplied by the user.
        assert (
            reply.index(f"#{expected[0]['id']} ·")
            < reply.index(f"#{expected[1]['id']} ·")
        )

        # Never include an unrelated task.
        assert unrelated["title"] not in reply
        assert "**Your tasks (3)**" not in reply

    # Ordinary "show all" requests still work.
    all_tasks = run_turn(
        "Show all tasks",
        **args,
    )

    assert "**Your tasks (3)**" in all_tasks
    assert first["title"] in all_tasks
    assert second["title"] in all_tasks
    assert unrelated["title"] in all_tasks

    # Read-only operations must not change records.
    assert len(list_tasks(db)) == 3



def test_mixed_update_and_read_in_one_turn(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": [
            "get_task",
            "update_task",
        ]
    }))

    edited = create_task(
        title="QA Mixed Edited",
        path=db,
    )

    inspected = create_task(
        title="QA Mixed Inspected",
        path=db,
    )

    untouched = create_task(
        title="QA Mixed Untouched",
        path=db,
    )

    class PlannedActions:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            return ActionPlan(operations=[
                Operation(
                    action="update_task",
                    task_id=edited["id"],
                    status="In progress",
                ),
                Operation(
                    action="get_task",
                    task_id=inspected["id"],
                ),
            ])

    planner = PlannedActions()

    answer = run_turn(
        f"Move task #{edited['id']} to In progress, "
        f"then show details for task #{inspected['id']}.",
        planner=planner,
        thread_id="mixed-actions-regression",
        db_path=db,
        checkpoint_path=cp,
        permissions_path=permissions,
        chroma_path=tmp_path / "chroma",
    )

    # Both operations must execute from one plan.
    assert planner.calls == 1

    assert answer.count(
        "✅ **Task updated**"
    ) == 1

    assert answer.count(
        "✅ **Task details**"
    ) == 1

    # Operations must be displayed in their planned order.
    assert (
        answer.index(f"#{edited['id']} ·")
        < answer.index(f"#{inspected['id']} ·")
    )

    # Only the requested update should change state.
    assert get_task(
        edited["id"], db
    )["status"] == "In progress"

    assert get_task(
        inspected["id"], db
    )["status"] == "To do"

    assert get_task(
        untouched["id"], db
    )["status"] == "To do"

    # No tasks were accidentally created or deleted.
    assert len(list_tasks(db)) == 3


def test_task_advice_respects_requested_count(tmp_path):
    import re

    db = tmp_path / "tasks.sqlite3"
    cp = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(json.dumps({
        "enabled_tools": ["list_tasks"]
    }))

    for number in range(1, 7):
        create_task(
            title=f"QA Advice Task {number}",
            priority="High" if number <= 2 else "Medium",
            path=db,
        )

    create_task(
        title="QA Completed Advice Task",
        status="Done",
        path=db,
    )

    before = list_tasks(db)

    class AdvicePlanner:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return ActionPlan(mode="task_advice")

    planner = AdvicePlanner()

    args = {
        "planner": planner,
        "thread_id": "advice-count-regression",
        "db_path": db,
        "checkpoint_path": cp,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    def recommendation_count(reply):
        return len(re.findall(
            r"(?m)^\d+\. \*\*#\d+",
            reply,
        ))

    default = run_turn(
        "Help me decide what to work on next. "
        "Do not change any tasks.",
        **args,
    )

    assert recommendation_count(default) == 3

    five = run_turn(
        "I'm feeling overwhelmed. "
        "Which five should I focus on first, and why? "
        "Please don't modify anything.",
        **args,
    )

    assert recommendation_count(five) == 5

    numeric = run_turn(
        "Recommend 5 tasks to focus on next. "
        "Don't modify anything.",
        **args,
    )

    assert recommendation_count(numeric) == 5

    # Completed tasks must not be recommended.
    assert "QA Completed Advice Task" not in five

    # Advice must never mutate task records.
    assert list_tasks(db) == before

    assert planner.calls == 3


def test_advice_ranking_balances_urgency_and_priority():
    import re

    from task_agent.response_formatting import (
        recommend_next_tasks,
    )

    def task(
        task_id,
        priority,
        due_date=None,
        status="To do",
    ):
        return {
            "id": task_id,
            "title": f"QA Ranking Task {task_id}",
            "priority": priority,
            "due_date": due_date,
            "status": status,
        }

    tasks = [
        task(1, "Medium", "2026-11-19"),
        task(2, "High"),
        task(3, "Medium", "2026-11-04"),
        task(4, "Low", "2026-10-29"),
        task(5, "Low", "2026-10-27"),
        task(6, "High", "2026-10-01", "Done"),
        task(7, "Medium", status="In progress"),
        task(8, "High", "2026-11-05"),
        task(9, "High", "2026-10-28"),
        task(10, "Low", "2026-11-04"),
    ]

    before = [dict(item) for item in tasks]

    result = recommend_next_tasks(
        tasks,
        today=date(2026, 10, 28),
        limit=10,
    )

    recommended_ids = [
        int(task_id)
        for task_id in re.findall(
            r"(?m)^\d+\. \*\*#(\d+) ·",
            result,
        )
    ]

    assert recommended_ids == [
        5,   # Overdue
        9,   # Due today
        4,   # Due tomorrow
        3,   # Due in seven days, Medium
        10,  # Due in seven days, Low
        8,   # Distant deadline, High
        2,   # No deadline, High
        1,   # Distant deadline, Medium
        7,   # No deadline, Medium
    ]

    # Completed task #6 must not appear.
    assert 6 not in recommended_ids

    # Recommendations must never change task data.
    assert tasks == before
