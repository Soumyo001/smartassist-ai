
import json

from task_agent.agent import (
    ActionPlan,
    Operation,
    run_turn,
)

from task_agent.storage import (
    create_task,
    list_tasks,
)

from task_agent.task_intelligence import (
    audit_last_priority_list,
)


class NoLLM:
    def invoke(self, messages):
        raise AssertionError(
            "This request should not call Groq"
        )


class DeletePlanner:
    def invoke(self, messages):
        return ActionPlan(
            mode="actions",
            operations=[
                Operation(
                    action="delete_task",
                    target="Keep this task",
                )
            ],
        )


class BadRequestError(Exception):
    def __init__(self):
        super().__init__("json_validate_failed")
        self.body = {
            "error": {
                "code": "json_validate_failed"
            }
        }


class FailingPlanner:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        raise BadRequestError()


def environment(tmp_path):

    policy = tmp_path / "permissions.json"

    policy.write_text(json.dumps({
        "enabled_tools": [
            "create_task",
            "get_task",
            "list_tasks",
            "update_task",
            "complete_task",
            "delete_task",
            "search_tasks",
        ]
    }))

    return {
        "db_path": tmp_path / "tasks.sqlite3",
        "checkpoint_path": tmp_path / "checkpoint.sqlite3",
        "chroma_path": tmp_path / "chroma",
        "permissions_path": policy,
        "thread_id": "reliability-test",
    }


def test_merge_cannot_delete_records(tmp_path):

    args = environment(tmp_path)

    source = create_task(
        title="Submit assignment",
        path=args["db_path"],
    )

    target = create_task(
        title="Submit assignment",
        path=args["db_path"],
    )

    result = run_turn(
        f"Merge #{source['id']} with #{target['id']} "
        "and keep the second one",
        planner=NoLLM(),
        **args,
    )

    assert "not changed or deleted" in result

    assert len(list_tasks(
        args["db_path"]
    )) == 2


def test_model_cannot_infer_deletion(tmp_path):

    args = environment(tmp_path)

    create_task(
        title="Keep this task",
        path=args["db_path"],
    )

    result = run_turn(
        "Please tidy up the duplicate tasks",
        planner=DeletePlanner(),
        **args,
    )

    assert "Could not delete_task" in result
    assert len(list_tasks(args["db_path"])) == 1


def test_order_audit_uses_actual_rows():

    correct = (
        "*Sorted by priority: Low → High → Medium*\n\n"
        "**Your tasks (3)**\n\n"
        "- **#1 · A**  \n"
        "  To do · Low priority · No due date\n"
        "- **#2 · B**  \n"
        "  To do · High priority · No due date\n"
        "- **#3 · C**  \n"
        "  To do · Medium priority · No due date"
    )

    assert "**Yes.**" in audit_last_priority_list(
        [correct]
    )


def test_order_audit_detects_wrong_order():

    incorrect = (
        "*Sorted by priority: Low → High → Medium*\n\n"
        "**Your tasks (2)**\n\n"
        "- **#1 · A**  \n"
        "  To do · High priority · No due date\n"
        "- **#2 · B**  \n"
        "  To do · Low priority · No due date"
    )

    assert "**No.**" in audit_last_priority_list(
        [incorrect]
    )


def test_groq_json_failure_is_bounded(tmp_path):

    args = environment(tmp_path)
    planner = FailingPlanner()

    response = run_turn(
        "Help me prioritize today's workload",
        planner=planner,
        **args,
    )

    assert planner.calls == 2
    assert "No task changes were made" in response
    assert list_tasks(args["db_path"]) == []
