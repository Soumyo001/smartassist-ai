"""Task-manager responses, filtering and sorting."""

import re


INTRO = (
    "Hi! I'm **SmartAssist AI**, your task manager. "
    "I can organize your tasks, update priorities and deadlines, "
    "find related work, and mark tasks complete. "
    "What would you like me to take care of?"
)


def normal_words(message: str) -> str:
    return " ".join(
        re.findall(r"\w+", message.casefold())
    )


def no_action_reply(message: str) -> str:
    text = normal_words(message)

    if text in {
        "hi",
        "hello",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "salam",
        "assalamualaikum",
    }:
        return INTRO

    if text in {
        "who are you",
        "what can you do",
        "help",
        "help me",
        "what do you do",
    }:
        return (
            "I'm **SmartAssist AI**, your task-management assistant. "
            "I can create, find, edit, prioritize, complete and delete tasks. "
            "You can also ask me to sort tasks by number, priority or due date. "
            "For example: **'Show my high-priority unfinished tasks, "
            "ordered by due date.'**"
        )

    if text in {
        "thanks",
        "thank you",
        "thx",
        "ok thanks",
    }:
        return (
            "You're welcome. "
            "What would you like me to organize next?"
        )

    if re.search(
        r"\b(create|make|add)\b.*"
        r"\b(tasks?|to dos?)\b",
        text,
    ):
        return (
            "Of course. **What should I name the task?** "
            "You can also tell me its priority and deadline."
        )

    if re.search(
        r"\b(change|update|move|finish|"
        r"complete|delete|remove)\b.*"
        r"\b(tasks?|one)\b",
        text,
    ):
        return (
            "Which task should I update? "
            "Tell me its name or ID, for example **#8**."
        )

    return (
        "I focus on **task-management**, "
        "so I can't do unrelated requests. "
        "Tell me what task you'd like me to create, "
        "find, update or complete, and I'll help you."
    )


def priority_requested(message: str) -> str | None:
    text = normal_words(message)

    patterns = {
        "High": r"\b(high|highs|highest|urgent)\b",
        "Medium": r"\b(medium|mediums)\b",
        "Low": r"\b(low|lows|lowest)\b",
    }

    found = [
        priority
        for priority, pattern in patterns.items()
        if re.search(pattern, text)
    ]

    # Multiple priorities usually indicate sorting,
    # rather than filtering to one priority.
    return found[0] if len(found) == 1 else None


def filter_tasks(
    tasks: list[dict],
    *,
    unfinished: bool = False,
    priority: str | None = None,
) -> list[dict]:

    return [
        task for task in tasks
        if (
            not unfinished
            or task["status"] != "Done"
        )
        and (
            priority is None
            or task["priority"] == priority
        )
    ]


def ordering(
    message: str,
    suggested: str | None = None,
) -> tuple[str, bool, list[str]]:
    """
    Return:
      sort_field, reverse, priority_order
    """

    text = normal_words(message)

    priority_order = [
        "High",
        "Low",
        "Medium",
    ]

    if re.search(
        r"\bhigh\b.*\bmedium\b.*\blow\b",
        text,
    ):
        priority_order = [
            "High",
            "Medium",
            "Low",
        ]

    elif re.search(
        r"\blow\b.*\bmedium\b.*\bhigh\b",
        text,
    ):
        priority_order = [
            "Low",
            "Medium",
            "High",
        ]

    if (
        "priority" in text
        and any(
            word in text
            for word in (
                "order", "sort", "rank",
                "first", "last",
            )
        )
    ):
        field = "priority"

    elif (
        any(
            word in text
            for word in (
                "due date",
                "deadline",
                "due first",
            )
        )
        and any(
            word in text
            for word in (
                "order",
                "sort",
                "earliest",
                "latest",
                "first",
            )
        )
    ):
        field = "due_date"

    elif any(
        word in text
        for word in (
            "alphabetically",
            "by name",
            "by title",
        )
    ):
        field = "title"

    elif any(
        word in text
        for word in (
            "by status",
            "by progress",
        )
    ):
        field = "status"

    elif re.search(
        r"\b(by id|by number|numerically|"
        r"ascending|descending|ordered|"
        r"sorted|sort|order)\b",
        text,
    ):
        field = "id"

    else:
        field = (
            suggested
            if suggested in {
                "id", "priority", "due_date",
                "title", "status",
            }
            else "id"
        )

    reverse = bool(
        re.search(
            r"\b(descending|reverse|newest|"
            r"largest first)\b",
            text,
        )
    )

    return field, reverse, priority_order


def sort_tasks(
    tasks: list[dict],
    *,
    field: str = "id",
    reverse: bool = False,
    priority_order: list[str] | None = None,
) -> list[dict]:

    priorities = priority_order or [
        "High",
        "Low",
        "Medium",
    ]

    priority_rank = {
        priority: index
        for index, priority in enumerate(priorities)
    }

    status_rank = {
        "To do": 0,
        "In progress": 1,
        "Done": 2,
    }

    items = list(tasks)

    if field == "priority":
        return sorted(
            items,
            key=lambda t: (
                priority_rank.get(
                    t["priority"], 99
                ),
                t["id"],
            ),
            reverse=reverse,
        )

    if field == "due_date":
        dated = [
            t for t in items
            if t.get("due_date")
        ]

        undated = [
            t for t in items
            if not t.get("due_date")
        ]

        # Undated tasks always appear last.
        return sorted(
            dated,
            key=lambda t: (
                t["due_date"],
                t["id"],
            ),
            reverse=reverse,
        ) + sorted(
            undated,
            key=lambda t: t["id"],
        )

    if field == "title":
        return sorted(
            items,
            key=lambda t: (
                t["title"].casefold(),
                t["id"],
            ),
            reverse=reverse,
        )

    if field == "status":
        return sorted(
            items,
            key=lambda t: (
                status_rank.get(
                    t["status"], 99
                ),
                t["id"],
            ),
            reverse=reverse,
        )

    return sorted(
        items,
        key=lambda t: t["id"],
        reverse=reverse,
    )


def format_task_list(
    tasks: list[dict],
    heading: str = "Your tasks",
) -> str:

    if not tasks:
        return (
            f"**{heading}**\n\n"
            "No matching tasks found."
        )

    lines = [
        f"**{heading} ({len(tasks)})**",
        "",
    ]

    for task in tasks:
        due = (
            task.get("due_date")
            or "No due date"
        )

        lines.append(
            f"- **#{task['id']} · {task['title']}**  \n"
            f"  {task['status']} · "
            f"{task['priority']} priority · {due}"
        )

    return "\n".join(lines)


def format_action_result(
    action: str,
    task: dict,
) -> str:

    labels = {
        "create_task": "Created task",
        "update_task": "Task updated",
        "complete_task": "Task completed",
        "delete_task": "Task deleted",
        "get_task": "Task details",
    }

    label = labels.get(
        action,
        "Task action completed",
    )

    due = (
        task.get("due_date")
        or "No due date"
    )

    return (
        f"✅ **{label}**\n\n"
        f"**#{task['id']} · {task['title']}**  \n"
        f"{task['status']} · "
        f"{task['priority']} priority · {due}"
        f"  \n*Action: `{action}`*"
    )