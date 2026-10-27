
"""Presentation helpers for SmartAssist's task-only responses."""

import re


INTRO = (
    "Hi! I'm **SmartAssist AI**, your task manager. "
    "I can create, find, prioritize, update, complete, and delete tasks. "
    "What would you like to organize?"
)


def no_action_reply(message: str) -> str:
    """Safe, helpful responses for greetings and non-task requests."""

    text = " ".join(message.casefold().strip().split())

    if text in {
        "hi", "hello", "hey", "hey there",
        "good morning", "good afternoon", "good evening",
        "assalamualaikum", "salam",
    }:
        return INTRO

    if text in {
        "who are you?", "who are you",
        "what can you do?", "what can you do",
        "help", "help me",
    }:
        return (
            "I'm **SmartAssist AI**, a task-management assistant. "
            "I can create tasks, search existing tasks, change priority or status, "
            "and complete or delete tasks through your task board. "
            "Try: **'Show my high-priority unfinished tasks.'**"
        )

    if text in {
        "thanks", "thank you", "thank you!",
        "thx", "ok thanks",
    }:
        return (
            "You're welcome! Tell me whenever you need "
            "help organizing your tasks."
        )

    return (
        "I'm here to help manage your tasks, rather than answer unrelated requests. "
        "If you meant a task instruction, could you explain what you'd like me to "
        "**create, find, change, complete, or delete**?"
    )


def priority_requested(message: str) -> str | None:
    """Detect explicit priority filters for listing tasks."""

    text = message.casefold()

    if re.search(
        r"\b(high[ -]priority|highest[ -]priority|urgent)\b",
        text
    ):
        return "High"

    if re.search(r"\b(medium[ -]priority)\b", text):
        return "Medium"

    if re.search(r"\b(low[ -]priority)\b", text):
        return "Low"

    return None


def filter_tasks(
    tasks: list[dict],
    *,
    unfinished: bool = False,
    priority: str | None = None,
) -> list[dict]:
    return [
        task for task in tasks
        if (not unfinished or task["status"] != "Done")
        and (priority is None or task["priority"] == priority)
    ]


def format_task_list(
    tasks: list[dict],
    heading: str = "Your tasks",
) -> str:

    if not tasks:
        return f"**{heading}**\n\nNo matching tasks found."

    lines = [f"**{heading} ({len(tasks)})**", ""]

    for task in tasks:
        due = task.get("due_date") or "No due date"

        lines.append(
            f"- **#{task['id']} · {task['title']}**  \n"
            f"  {task['status']} · {task['priority']} priority · {due}"
        )

    return "\n".join(lines)


def format_action_result(action: str, task: dict) -> str:

    titles = {
        "create_task": "Created task",
        "update_task": "Task updated",
        "complete_task": "Task completed",
        "delete_task": "Task deleted",
        "get_task": "Task details",
    }

    label = titles.get(action, "Task action completed")
    due = task.get("due_date") or "No due date"

    return (
        f"✅ **{label}**\n\n"
        f"**#{task['id']} · {task['title']}**  \n"
        f"{task['status']} · {task['priority']} priority · {due}"
        f"  \n*Action: `{action}`*"
    )
