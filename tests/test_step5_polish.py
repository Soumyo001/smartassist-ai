from task_agent.response_formatting import (
    filter_tasks,
    format_action_result,
    format_task_list,
    no_action_reply,
    priority_requested,
)


def test_greetings_and_off_topic():

    assert "SmartAssist AI" in no_action_reply("hi")

    assert "task-management" in no_action_reply(
        "who are you?"
    )

    assert "unrelated" in no_action_reply(
        "write a poem"
    )


def test_priority_filter_is_exact():

    tasks = [
        {
            "id": 1,
            "title": "A",
            "priority": "High",
            "status": "To do",
        },
        {
            "id": 2,
            "title": "B",
            "priority": "Low",
            "status": "To do",
        },
        {
            "id": 3,
            "title": "C",
            "priority": "High",
            "status": "Done",
        },
    ]

    assert priority_requested(
        "What are my high priority tasks?"
    ) == "High"

    matches = filter_tasks(
        tasks,
        priority="High",
        unfinished=True,
    )

    assert [
        task["id"] for task in matches
    ] == [1]

    assert "#1" in format_task_list(matches)
    assert "#2" not in format_task_list(matches)


def test_action_output_contains_actual_fields():

    text = format_action_result(
        "update_task",
        {
            "id": 7,
            "title": "Demo",
            "status": "In progress",
            "priority": "Medium",
            "due_date": "2026-10-28",
        }
    )

    assert "Task updated" in text
    assert "In progress" in text
    assert "2026-10-28" in text