from task_agent.agent import normalize_task_datetime


def test_datetime_is_normalized_without_losing_time():

    result = normalize_task_datetime({
        "title": "Attend exam",
        "description": "",
        "due_date": "2026-10-27T17:00:00",
    })

    assert result["due_date"] == "2026-10-27"

    assert (
        "5:00 PM (Asia/Dhaka)"
        in result["description"]
    )


def test_date_only_remains_unchanged():

    original = {
        "title": "Finish presentation",
        "description": "Complete slides",
        "due_date": "2026-10-28",
    }

    result = normalize_task_datetime(original)

    assert result == original