import chromadb

from .storage import DATA_DIR


def get_task_collection():

    path = DATA_DIR / "chroma"

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(path)
    )

    return client.get_or_create_collection(
        name="taskflow_tasks"
    )