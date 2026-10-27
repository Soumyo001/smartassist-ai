import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from . import storage
from . import vector_store

from .tool_policy import (
    PERMISSIONS_PATH,
    get_enabled_tool_names,
    require_allowed,
)


def build_allowed_tools(
    db_path: Path = storage.DB_PATH,
    permissions_path: Path = PERMISSIONS_PATH,
    chroma_path: Path | None = None
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
        """Mark a task as Done using its integer ID."""

        check("complete_task")

        return json.dumps(
            storage.complete_task(
                task_id,
                path=db_path
            )
        )

    @tool("delete_task")
    def delete_task_tool(task_id: int) -> str:
        """Delete a task using its resolved, unambiguous ID."""

        check("delete_task")

        return json.dumps(
            storage.delete_task(
                task_id,
                path=db_path
            )
        )

    @tool("search_tasks")
    def search_tasks_tool(
        query: str,
        status: Literal[
            "To do", "In progress", "Done"
        ] | None = None,
        limit: int = 5,
    ) -> str:
        """
        Find related tasks using semantic similarity.

        Use this when the user references a task by
        topic, meaning, description, or approximate name.

        For unfinished tasks, search both To do and
        In progress, or search without a status filter
        and exclude Done.

        Search results contain stable task IDs.
        A search match is not permission to modify
        an ambiguous task.
        """

        check("search_tasks")

        matches = vector_store.search_tasks(
            query=query,
            status=status,
            limit=limit,
            db_path=db_path,
            chroma_path=chroma_path
        )

        return json.dumps(matches)

    registry = {
        t.name: t
        for t in (
            create_task_tool,
            list_tasks_tool,
            get_task_tool,
            update_task_tool,
            complete_task_tool,
            delete_task_tool,
            search_tasks_tool,
        )
    }

    enabled = get_enabled_tool_names(
        permissions_path
    )

    return [
        registry[name]
        for name in sorted(enabled)
    ]