
import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from . import storage
from .tool_policy import (
    PERMISSIONS_PATH,
    get_enabled_tool_names,
    require_allowed,
)


def build_allowed_tools(
    db_path: Path = storage.DB_PATH,
    permissions_path: Path = PERMISSIONS_PATH
):
    def check(name):
        require_allowed(name, permissions_path)

    @tool("create_task")
    def create_task_tool(
        title: str,
        description: str = "",
        status: Literal[
            "To do", "In progress", "Done"
        ] = "To do",
        priority: Literal[
            "Low", "Medium", "High"
        ] = "Medium",
        due_date: str | None = None,
    ) -> str:
        """Create a task. due_date uses YYYY-MM-DD or null."""

        check("create_task")

        result = storage.create_task(
            title=title,
            description=description,
            status=status,
            priority=priority,
            due_date=due_date,
            path=db_path,
        )

        return json.dumps(result)

    @tool("list_tasks")
    def list_tasks_tool(
        status: Literal[
            "To do", "In progress", "Done"
        ] | None = None
    ) -> str:
        """List saved tasks, optionally filtered by status."""

        check("list_tasks")

        return json.dumps(
            storage.list_tasks(
                path=db_path,
                status=status
            )
        )

    @tool("get_task")
    def get_task_tool(task_id: int) -> str:
        """Retrieve one task using its integer task ID."""

        check("get_task")

        return json.dumps(
            storage.get_task(task_id, db_path)
        )

    @tool("update_task")
    def update_task_tool(
        task_id: int,
        title: str | None = None,
        description: str | None = None,
        status: Literal[
            "To do", "In progress", "Done"
        ] | None = None,
        priority: Literal[
            "Low", "Medium", "High"
        ] | None = None,
        due_date: str | None = None,
        clear_due_date: bool = False,
    ) -> str:
        """Update a task identified by ID. Use clear_due_date to remove its date."""

        check("update_task")

        changes = {
            k: v
            for k, v in {
                "title": title,
                "description": description,
                "status": status,
                "priority": priority,
                "due_date": due_date,
            }.items()
            if v is not None
        }

        if clear_due_date:
            if due_date is not None:
                raise ValueError(
                    "Cannot set and clear due_date simultaneously"
                )

            changes["due_date"] = None

        result = storage.update_task(
            task_id,
            path=db_path,
            **changes
        )

        return json.dumps(result)

    @tool("complete_task")
    def complete_task_tool(task_id: int) -> str:
        """Mark the task with this integer ID as Done."""

        check("complete_task")

        return json.dumps(
            storage.complete_task(task_id, path=db_path)
        )

    @tool("delete_task")
    def delete_task_tool(task_id: int) -> str:
        """Delete a task using its resolved, unambiguous integer ID."""

        check("delete_task")

        return json.dumps(
            storage.delete_task(task_id, path=db_path)
        )

    registry = {
        t.name: t
        for t in (
            create_task_tool,
            list_tasks_tool,
            get_task_tool,
            update_task_tool,
            complete_task_tool,
            delete_task_tool,
        )
    }

    enabled = get_enabled_tool_names(
        permissions_path
    )

    return [
        registry[name]
        for name in sorted(enabled)
    ]
