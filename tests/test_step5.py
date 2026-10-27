
import json

from task_agent.agent import (
    ActionPlan,
    Operation,
    run_turn,
)

from task_agent.chat_history import load_chat_history


class FakePlanner:
    def invoke(self, messages):
        return ActionPlan(
            operations=[
                Operation(
                    action="create_task",
                    title="Prepare demo",
                    priority="High",
                )
            ]
        )


def test_chat_history_survives_restart(tmp_path):

    db = tmp_path / "tasks.sqlite3"
    checkpoint = tmp_path / "checkpoints.sqlite3"
    chroma = tmp_path / "chroma"
    policy = tmp_path / "permissions.json"

    policy.write_text(
        json.dumps({
            "enabled_tools": ["create_task"]
        })
    )

    response = run_turn(
        "Create a high-priority task to prepare the demo",
        thread_id="test-web-ui",
        planner=FakePlanner(),
        db_path=db,
        checkpoint_path=checkpoint,
        permissions_path=policy,
        chroma_path=chroma,
    )

    assert "Created" in response

    # Opens a new checkpoint connection,
    # simulating a browser refresh or app restart.
    history = load_chat_history(
        "test-web-ui",
        checkpoint_path=checkpoint,
        db_path=db,
        permissions_path=policy,
        chroma_path=chroma,
    )

    assert [
        message["role"]
        for message in history
    ] == ["user", "assistant"]

    assert "prepare the demo" in (
        history[0]["content"].lower()
    )

    assert "Created" in history[1]["content"]

    # Another thread must not inherit this conversation.
    assert load_chat_history(
        "unknown-thread",
        checkpoint_path=checkpoint,
        db_path=db,
        permissions_path=policy,
        chroma_path=chroma,
    ) == []
