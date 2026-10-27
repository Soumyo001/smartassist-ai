import json

import pytest

from task_agent.storage import (
    create_task,
    update_task,
    delete_task,
)

from task_agent.vector_store import (
    get_task_collection,
    sync_task_index,
    search_tasks,
)

from task_agent.tools import build_allowed_tools


def test_semantic_search_and_index_lifecycle(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    chroma = tmp_path / "chroma"

    documentation = create_task(
        title="Update API authentication documentation",
        description=(
            "Write documentation for authentication "
            "endpoints and developer integration."
        ),
        priority="High",
        path=db,
    )

    create_task(
        title="Buy groceries",
        description="Purchase milk, eggs and vegetables.",
        priority="Low",
        path=db,
    )

    # Index SQLite tasks into ChromaDB.
    result = sync_task_index(
        db_path=db,
        chroma_path=chroma
    )

    assert result["sqlite_tasks"] == 2
    assert result["chroma_tasks"] == 2

    # Re-indexing unchanged tasks should do nothing.
    repeated = sync_task_index(
        db_path=db,
        chroma_path=chroma
    )

    assert repeated["indexed_or_updated"] == 0

    # Semantic search should find the relevant task.
    matches = search_tasks(
        "API authentication documentation",
        db_path=db,
        chroma_path=chroma
    )

    assert matches
    assert matches[0]["id"] == documentation["id"]

    # Change status and priority in SQLite.
    update_task(
        documentation["id"],
        status="Done",
        priority="Low",
        path=db
    )

    # Search automatically synchronizes the index.
    unfinished = search_tasks(
        "API authentication documentation",
        status="To do",
        db_path=db,
        chroma_path=chroma
    )

    assert all(
        task["id"] != documentation["id"]
        for task in unfinished
    )

    # Verify ChromaDB metadata was updated.
    collection = get_task_collection(chroma)

    indexed = collection.get(
        ids=[str(documentation["id"])],
        include=["metadatas"]
    )

    assert indexed["metadatas"][0]["status"] == "Done"
    assert indexed["metadatas"][0]["priority"] == "Low"

    # Delete from SQLite.
    delete_task(
        documentation["id"],
        path=db
    )

    # Search should remove the stale vector record.
    search_tasks(
        "API authentication documentation",
        db_path=db,
        chroma_path=chroma
    )

    persisted = get_task_collection(chroma)

    assert persisted.count() == 1

    deleted = persisted.get(
        ids=[str(documentation["id"])]
    )

    assert deleted["ids"] == []


def test_search_tool_respects_allowlist(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    chroma = tmp_path / "chroma"
    policy = tmp_path / "permissions.json"

    create_task(
        title="Review project documentation",
        description="Review technical implementation guides.",
        path=db
    )

    policy.write_text(
        json.dumps({
            "enabled_tools": ["search_tasks"]
        }),
        encoding="utf-8"
    )

    tools = build_allowed_tools(
        db_path=db,
        permissions_path=policy,
        chroma_path=chroma
    )

    assert [tool.name for tool in tools] == [
        "search_tasks"
    ]

    search_tool = tools[0]

    result = json.loads(
        search_tool.invoke({
            "query": "technical documentation"
        })
    )

    assert result
    assert result[0]["title"] == (
        "Review project documentation"
    )

    # Revoke permission after the tool was created.
    policy.write_text(
        json.dumps({
            "enabled_tools": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(PermissionError):
        search_tool.invoke({
            "query": "technical documentation"
        })