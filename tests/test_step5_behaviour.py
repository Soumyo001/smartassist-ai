"""Regression tests for SmartAssist's task intelligence."""

import json

from task_agent.agent import (
    ActionPlan,
    Operation,
    run_turn,
)

from task_agent.storage import (
    create_task,
    get_task,
    list_tasks,
)


class Planner:
    def __init__(self, *plans):
        self.plans = list(plans)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        if not self.plans:
            raise AssertionError(
                "Unexpected model call"
            )

        return self.plans.pop(0)


def setup(tmp_path):

    policy = tmp_path / "allowlist.json"

    policy.write_text(
        json.dumps({
            "enabled_tools": [
                "create_task",
                "list_tasks",
                "get_task",
                "update_task",
                "complete_task",
                "delete_task",
                "search_tasks",
            ]
        })
    )

    return {
        "db_path": tmp_path / "tasks.sqlite3",
        "checkpoint_path": (
            tmp_path / "checkpoints.sqlite3"
        ),
        "chroma_path": tmp_path / "chroma",
        "permissions_path": policy,
        "thread_id": "behavior-test",
    }


def test_ordered_means_ascending_ids(tmp_path):

    args = setup(tmp_path)

    create_task(
        title="First",
        path=args["db_path"],
    )

    create_task(
        title="Second",
        path=args["db_path"],
    )

    response = run_turn(
        "show tasks ordered",
        planner=Planner(),
        **args,
    )

    assert response.index("#1") < response.index("#2")
    assert response.count("Your tasks") == 1


def test_duplicate_titles_get_suffix(tmp_path):

    args = setup(tmp_path)

    create_task(
        title="Submit assignment",
        path=args["db_path"],
    )

    planner = Planner(
        ActionPlan(
            operations=[
                Operation(
                    action="create_task",
                    title="Submit assignment",
                    priority="Low",
                )
            ]
        )
    )

    response = run_turn(
        "Create another low-priority Submit assignment task",
        planner=planner,
        **args,
    )

    assert "Submit assignment (2)" in response

    titles = [
        task["title"]
        for task in list_tasks(
            args["db_path"]
        )
    ]

    assert sorted(titles) == sorted([
        "Submit assignment",
        "Submit assignment (2)",
    ])


def test_new_instruction_replaces_pending_edit(tmp_path):

    args = setup(tmp_path)

    first = create_task(
        title="Submit assignment",
        path=args["db_path"],
    )

    second = create_task(
        title="Submit assignment",
        path=args["db_path"],
    )

    planner = Planner(
        ActionPlan(
            operations=[
                Operation(
                    action="update_task",
                    target="Submit assignment",
                    due_date="2026-11-06",
                )
            ]
        )
    )

    question = run_turn(
        "Set the assignment due date to November 6",
        planner=planner,
        **args,
    )

    assert "Which one" in question

    response = run_turn(
        f"What's the priority of ${second['id']} task? "
        "I want the medium ones to set",
        planner=planner,
        **args,
    )

    assert "Medium priority" in response

    assert get_task(
        second["id"],
        args["db_path"],
    )["priority"] == "Medium"

    # The old pending due-date change must NOT execute.
    assert get_task(
        first["id"],
        args["db_path"],
    )["due_date"] is None

    assert get_task(
        second["id"],
        args["db_path"],
    )["due_date"] is None

    # Second instruction was handled deterministically.
    assert planner.calls == 1


def test_hash_id_resumes_correct_pending_task(tmp_path):

    args = setup(tmp_path)

    create_task(
        title="Review report",
        path=args["db_path"],
    )

    second = create_task(
        title="Review report",
        path=args["db_path"],
    )

    planner = Planner(
        ActionPlan(
            operations=[
                Operation(
                    action="update_task",
                    target="Review report",
                    priority="High",
                )
            ]
        )
    )

    run_turn(
        "Make Review report high priority",
        planner=planner,
        **args,
    )

    response = run_turn(
        f"#{second['id']} one",
        planner=planner,
        **args,
    )

    assert "Task updated" in response

    assert get_task(
        second["id"],
        args["db_path"],
    )["priority"] == "High"


def test_vague_create_request_asks_for_name(tmp_path):

    args = setup(tmp_path)

    reply = run_turn(
        "can you create task now?",
        planner=Planner(),
        **args,
    )

    assert "name the task" in reply.lower()

    assert list_tasks(
        args["db_path"]
    ) == []