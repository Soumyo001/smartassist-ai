import json

from task_agent.agent import (
    ActionPlan,
    run_turn,
)


class FakePlanner:
    def invoke(self, messages):
        return ActionPlan(
            mode="conversation",
            message=(
                "I'm SmartAssist AI, ready "
                "to help organize your work."
            ),
        )


def test_progress_events_and_response(tmp_path):

    permissions = tmp_path / "permissions.json"

    permissions.write_text(
        json.dumps({
            "enabled_tools": ["list_tasks"]
        })
    )

    progress_events = []

    response = run_turn(
        "How can you help organize my day?",
        thread_id="progress-test",
        planner=FakePlanner(),
        on_progress=progress_events.append,
        db_path=tmp_path / "tasks.sqlite3",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        chroma_path=tmp_path / "chroma",
        permissions_path=permissions,
    )

    assert "SmartAssist AI" in response

    assert any(
        "Understanding your request" in event
        for event in progress_events
    )

    assert "Response ready." in progress_events