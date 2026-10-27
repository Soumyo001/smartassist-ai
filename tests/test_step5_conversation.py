
"""Tests for priority sorting and natural task conversation."""

import json

from task_agent.agent import ActionPlan, run_turn
from task_agent.storage import create_task, list_tasks

from task_agent.response_formatting import (
    ordering,
    sort_tasks,
    recommend_next_tasks,
)


class FakePlanner:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.plan


def setup_agent(tmp_path):
    policy = tmp_path / "allowlist.json"

    policy.write_text(json.dumps({
        "enabled_tools": ["list_tasks"]
    }))

    return {
        "db_path": tmp_path / "tasks.sqlite3",
        "checkpoint_path": tmp_path / "checkpoints.sqlite3",
        "permissions_path": policy,
        "chroma_path": tmp_path / "chroma",
        "thread_id": "conversation-test",
    }


def test_default_priority_order():
    tasks = [
        {"id": 1, "priority": "Low"},
        {"id": 2, "priority": "Medium"},
        {"id": 3, "priority": "High"},
    ]

    field, reverse, ranks = ordering(
        "Show my tasks ordered by priority descending"
    )

    result = sort_tasks(
        tasks,
        field=field,
        reverse=reverse,
        priority_order=ranks,
    )

    assert [
        task["priority"] for task in result
    ] == ["High", "Medium", "Low"]


def test_custom_priority_order():
    tasks = [
        {"id": 1, "priority": "Medium"},
        {"id": 2, "priority": "Low"},
        {"id": 3, "priority": "High"},
    ]

    field, reverse, ranks = ordering(
        "Show my tasks ordered by priority "
        "where Low comes after High"
    )

    result = sort_tasks(
        tasks,
        field=field,
        reverse=reverse,
        priority_order=ranks,
    )

    assert [
        task["priority"] for task in result
    ] == ["High", "Low", "Medium"]


def test_natural_conversation(tmp_path):
    args = setup_agent(tmp_path)

    planner = FakePlanner(ActionPlan(
        mode="conversation",
        message=(
            "I'm ready to help you organize your day. "
            "How are things going?"
        ),
    ))

    response = run_turn(
        "How is your day going?",
        planner=planner,
        **args,
    )

    assert "organize your day" in response
    assert planner.calls == 1
    assert list_tasks(args["db_path"]) == []


def test_task_advice_uses_actual_database(tmp_path):
    args = setup_agent(tmp_path)

    create_task(
        title="Clean desk",
        priority="Low",
        path=args["db_path"],
    )

    create_task(
        title="Finish important demo",
        priority="High",
        path=args["db_path"],
    )

    planner = FakePlanner(ActionPlan(
        mode="task_advice",
        operations=[],
    ))

    response = run_turn(
        "I have too many tasks. What should I do first?",
        planner=planner,
        **args,
    )

    assert "2 unfinished" in response
    assert "#2" in response
    assert "#1" in response
    assert response.index("#2") < response.index("#1")

    # Advice must not change saved tasks.
    assert len(list_tasks(args["db_path"])) == 2
    assert all(
        task["status"] == "To do"
        for task in list_tasks(args["db_path"])
    )


def test_out_of_scope_request_is_declined(tmp_path):
    args = setup_agent(tmp_path)

    # Even if a model returns an inappropriate message,
    # the backend uses its fixed scope response.
    planner = FakePlanner(ActionPlan(
        mode="out_of_scope",
        message="Sure! Let's play a game.",
    ))

    response = run_turn(
        "Let's play a game",
        planner=planner,
        **args,
    )

    assert "task management" in response.lower()
    assert "Let's play a game" not in response