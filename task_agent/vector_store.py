from pathlib import Path

import chromadb

from . import storage
from .storage import DATA_DIR, DB_PATH


COLLECTION_NAME = "smartassist_tasks"

# Initial relevance cutoff calibrated using
# observed search distances in this project.
# Lower distance means stronger similarity.
MAX_SEARCH_DISTANCE = 1.20

VALID_STATUSES = {
    "To do",
    "In progress",
    "Done",
}


def get_task_collection(
    chroma_path: Path | None = None
):
    path = (
        Path(chroma_path)
        if chroma_path is not None
        else DATA_DIR / "chroma"
    )

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(path)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def task_to_document(task: dict) -> str:
    """
    Convert a task into searchable text.
    """

    title = task["title"].strip()
    description = (task["description"] or "").strip()

    if description:
        return (
            f"Task: {title}. "
            f"Description: {description}"
        )

    return f"Task: {title}"


def task_to_metadata(task: dict) -> dict:
    """
    Metadata supports exact filters.
    Empty dates are stored as empty strings.
    """

    return {
        "task_id": int(task["id"]),
        "title": task["title"],
        "status": task["status"],
        "priority": task["priority"],
        "due_date": task["due_date"] or "",
    }


def sync_task_index(
    db_path: Path = DB_PATH,
    chroma_path: Path | None = None
) -> dict:
    """
    Synchronize ChromaDB from SQLite.

    SQLite remains the source of truth.

    - New SQLite tasks are indexed.
    - Modified tasks are re-indexed.
    - Deleted SQLite tasks are removed.
    - Unchanged tasks are not re-embedded.
    """

    tasks = storage.list_tasks(path=db_path)

    collection = get_task_collection(chroma_path)

    existing = collection.get(
        include=["documents", "metadatas"]
    )

    existing_ids = existing["ids"]

    existing_documents = (
        existing.get("documents") or []
    )

    existing_metadatas = (
        existing.get("metadatas") or []
    )

    existing_by_id = {}

    for index, task_id in enumerate(existing_ids):
        existing_by_id[task_id] = {
            "document": existing_documents[index],
            "metadata": existing_metadatas[index],
        }

    sqlite_ids = set()

    upsert_ids = []
    upsert_documents = []
    upsert_metadatas = []

    for task in tasks:
        task_id = str(task["id"])

        sqlite_ids.add(task_id)

        document = task_to_document(task)
        metadata = task_to_metadata(task)

        previous = existing_by_id.get(task_id)

        if (
            previous is not None
            and previous["document"] == document
            and previous["metadata"] == metadata
        ):
            continue

        upsert_ids.append(task_id)
        upsert_documents.append(document)
        upsert_metadatas.append(metadata)

    if upsert_ids:
        collection.upsert(
            ids=upsert_ids,
            documents=upsert_documents,
            metadatas=upsert_metadatas
        )

    stale_ids = sorted(
        set(existing_ids) - sqlite_ids
    )

    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "sqlite_tasks": len(tasks),
        "indexed_or_updated": len(upsert_ids),
        "removed": len(stale_ids),
        "chroma_tasks": collection.count()
    }


def search_tasks(
    query: str,
    *,
    status: str | None = None,
    limit: int = 5,
    db_path: Path = DB_PATH,
    chroma_path: Path | None = None
) -> list[dict]:
    """
    Semantic search using ChromaDB.

    Returns current SQLite task records,
    ranked by vector similarity.

    Lower distance means a closer match.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Search query cannot be empty"
        )

    if status is not None and status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid task status: {status}"
        )

    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError(
            "limit must be between 1 and 50"
        )

    # Reconcile all changes from SQLite before searching.
    sync_task_index(
        db_path=db_path,
        chroma_path=chroma_path
    )

    collection = get_task_collection(chroma_path)

    total = collection.count()

    if total == 0:
        return []

    query_args = {
        "query_texts": [query.strip()],
        "n_results": min(limit, total),
        "include": ["distances"],
    }

    if status is not None:
        query_args["where"] = {
            "status": status
        }

    results = collection.query(**query_args)

    matched_ids = results["ids"][0]

    distances = (
        results.get("distances") or [[]]
    )[0]

    matches = []

    for task_id, distance in zip(
        matched_ids,
        distances
    ):
        # Nearest does not necessarily mean relevant.
        # Do not return weak semantic matches.
        if (
            distance is None
            or float(distance) > MAX_SEARCH_DISTANCE
        ):
            continue

        # Never trust the vector index as the source of truth.
        task = storage.get_task(
            int(task_id),
            path=db_path
        )

        if task is None:
            continue

        if (
            status is not None
            and task["status"] != status
        ):
            continue

        task["distance"] = float(distance)
        matches.append(task)

    return matches