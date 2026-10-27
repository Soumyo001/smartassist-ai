
import json
import pytest

from task_agent.response_formatting import (
    ordering,
    sort_tasks,
)

from task_agent.task_intelligence import (
    is_sort_correction,
    is_sort_audit_question,
)

from task_agent.agent import run_turn
from task_agent.storage import create_task


@pytest.mark.parametrize(
    "instruction,expected",
    [
        (
            "Show tasks ordered by priority",
            ["High", "Medium", "Low"],
        ),
        (
            "Show tasks ordered by priority descending",
            ["High", "Medium", "Low"],
        ),
        (
            "Show tasks ordered by priority ascending",
            ["Low", "Medium", "High"],
        ),
        (
            "Show all tasks where the high priority "
            "task comes after low priority tasks",
            ["Low", "High", "Medium"],
        ),
        (
            "Show tasks where Low comes after High",
            ["High", "Low", "Medium"],
        ),
        (
            "Show tasks where Medium comes before High",
            ["Medium", "High", "Low"],
        ),
    ],
)
def test_natural_priority_sorting(
    instruction, expected
):

    tasks = [
        {"id": 1, "priority": "Low"},
        {"id": 2, "priority": "High"},
        {"id": 3, "priority": "Medium"},
    ]

    field, reverse, priorities = ordering(
        instruction
    )

    result = sort_tasks(
        tasks,
        field=field,
        reverse=reverse,
        priority_order=priorities,
    )

    assert [
        task["priority"]
        for task in result
    ] == expected


def test_sort_correction_recognized():

    assert is_sort_correction(
        "earlier response was not ordered"
    )

    assert is_sort_correction(
        "that wasn't sorted correctly"
    )


def test_correction_replays_previous_list(tmp_path):

    db = tmp_path / "tasks.sqlite3"
    checkpoint = tmp_path / "checkpoints.sqlite3"
    permissions = tmp_path / "permissions.json"

    permissions.write_text(
        json.dumps({
            "enabled_tools": ["list_tasks"]
        })
    )

    create_task(
        title="Low-priority task",
        priority="Low",
        path=db,
    )

    create_task(
        title="High-priority task",
        priority="High",
        path=db,
    )

    create_task(
        title="Medium-priority task",
        priority="Medium",
        path=db,
    )

    class NoLLM:
        def invoke(self, messages):
            raise AssertionError(
                "Sorting should not call Groq"
            )

    arguments = {
        "thread_id": "sorting-test",
        "planner": NoLLM(),
        "db_path": db,
        "checkpoint_path": checkpoint,
        "permissions_path": permissions,
        "chroma_path": tmp_path / "chroma",
    }

    request = (
        "Show me all the tasks where the "
        "high priority task comes after "
        "low priority tasks"
    )

    first = run_turn(
        request,
        **arguments,
    )

    corrected = run_turn(
        "Earlier response was not ordered",
        **arguments,
    )

    for response in (first, corrected):
        assert (
            response.index("#1")
            < response.index("#2")
            < response.index("#3")
        )

        assert "Low → High → Medium" in response

def test_sort_correction_and_audit_are_distinct():

    correction = "Earlier response was not ordered"

    assert is_sort_correction(correction)
    assert not is_sort_audit_question(correction)

    question = "Was the earlier response ordered?"

    assert not is_sort_correction(question)
    assert is_sort_audit_question(question)

    negative_question = (
        "Was the earlier response not ordered?"
    )

    assert not is_sort_correction(negative_question)
    assert is_sort_audit_question(negative_question)
