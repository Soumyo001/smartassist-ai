import json

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from task_agent.agent import (
    ActionPlan,
    Operation,
    build_graph,
)

from task_agent.storage import (
    list_tasks,
    create_task,
)


class FakePlanner:

    def __init__(self, plans):
        self.plans = list(plans)
        self.called = 0

    def invoke(self, messages):
        self.called += 1
        return self.plans.pop(0)


def setup_environment(tmp_path, allowed):

    db = tmp_path / "tasks.sqlite3"
    checkpoint = tmp_path / "memory.sqlite3"
    chroma = tmp_path / "chroma"
    policy = tmp_path / "allowlist.json"

    policy.write_text(
        json.dumps({
            "enabled_tools": allowed
        }),
        encoding="utf-8",
    )

    return db, checkpoint, chroma, policy


def send_message(graph, text, thread="test"):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=text)
            ],
            "response": "",
            "results": [],
        },
        config={
            "configurable": {
                "thread_id": thread
            }
        },
    )

    return result["response"]


def test_multi_action_and_completion(tmp_path):

    db, cp, chroma, policy = setup_environment(
        tmp_path,
        [
            "create_task",
            "list_tasks",
            "get_task",
            "update_task",
            "complete_task",
            "search_tasks",
        ],
    )

    planner = FakePlanner([
        ActionPlan(operations=[
            Operation(
                action="create_task",
                title="Write report",
                priority="High",
            ),
            Operation(
                action="create_task",
                title="Review slides",
            ),
        ]),
        ActionPlan(operations=[
            Operation(
                action="complete_task",
                target="Write report",
            ),
        ]),
    ])

    with SqliteSaver.from_conn_string(
        str(cp)
    ) as saver:

        graph = build_graph(
            checkpointer=saver,
            planner=planner,
            db_path=db,
            permissions_path=policy,
            chroma_path=chroma,
        )

        result = send_message(
            graph,
            "Create Write report and Review slides"
        )

        assert "Created" in result

        result = send_message(
            graph,
            "Mark Write report done"
        )

        assert "complete_task" in result

    tasks = list_tasks(db)

    assert len(tasks) == 2

    report = next(
        task for task in tasks
        if task["title"] == "Write report"
    )

    assert report["status"] == "Done"


def test_ambiguity_survives_restart(tmp_path):

    db, cp, chroma, policy = setup_environment(
        tmp_path,
        [
            "list_tasks",
            "get_task",
            "update_task",
            "search_tasks",
        ],
    )

    create_task(
        title="Update API docs",
        path=db
    )

    create_task(
        title="Test API routes",
        path=db
    )

    planner = FakePlanner([
        ActionPlan(operations=[
            Operation(
                action="update_task",
                target="API",
                priority="Low",
            )
        ])
    ])

    with SqliteSaver.from_conn_string(
        str(cp)
    ) as saver:

        graph = build_graph(
            checkpointer=saver,
            planner=planner,
            db_path=db,
            permissions_path=policy,
            chroma_path=chroma,
        )

        response = send_message(
            graph,
            "Change the priority of the API task to Low"
        )

        assert "Which task" in response

        assert all(
            task["priority"] == "Medium"
            for task in list_tasks(db)
        )

    # Reopen the checkpoint in a separate context.
    with SqliteSaver.from_conn_string(
        str(cp)
    ) as saver:

        graph = build_graph(
            checkpointer=saver,
            planner=planner,
            db_path=db,
            permissions_path=policy,
            chroma_path=chroma,
        )

        response = send_message(
            graph,
            "second one"
        )

        assert "update_task" in response

    # The clarification should not invoke the LLM again.
    assert planner.called == 1

    assert sum(
        task["priority"] == "Low"
        for task in list_tasks(db)
    ) == 1


def test_disabled_delete_is_blocked(tmp_path):

    db, cp, chroma, policy = setup_environment(
        tmp_path,
        [
            "list_tasks",
            "get_task",
            "search_tasks",
        ],
    )

    task = create_task(
        title="Delete me",
        path=db
    )

    planner = FakePlanner([
        ActionPlan(operations=[
            Operation(
                action="delete_task",
                target="Delete me",
            )
        ])
    ])

    with SqliteSaver.from_conn_string(
        str(cp)
    ) as saver:

        graph = build_graph(
            checkpointer=saver,
            planner=planner,
            db_path=db,
            permissions_path=policy,
            chroma_path=chroma,
        )

        response = send_message(
            graph,
            "Delete 'Delete me'"
        )

    assert "Could not delete_task" in response

    # The database record must still exist.
    assert list_tasks(db)[0]["id"] == task["id"]
