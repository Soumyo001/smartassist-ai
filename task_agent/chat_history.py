
"""Load persistent conversation messages from LangGraph checkpoints."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .agent import build_graph
from .storage import DB_PATH, DATA_DIR
from .tool_policy import PERMISSIONS_PATH


def load_chat_history(
    thread_id: str,
    *,
    checkpoint_path: Path | None = None,
    db_path: Path = DB_PATH,
    permissions_path: Path = PERMISSIONS_PATH,
    chroma_path: Path | None = None,
) -> list[dict[str, str]]:
    """Read previous user/assistant turns without making an LLM request."""

    path = Path(
        checkpoint_path or DATA_DIR / "agent_checkpoints.sqlite3"
    )

    if not path.exists():
        return []

    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = build_graph(
            checkpointer=saver,
            planner=None,  # Reading state only; no agent execution.
            db_path=db_path,
            permissions_path=permissions_path,
            chroma_path=chroma_path,
        )

        snapshot = graph.get_state({
            "configurable": {
                "thread_id": thread_id
            }
        })

    result = []

    for message in snapshot.values.get("messages", []):

        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        content = message.content

        result.append({
            "role": role,
            "content": (
                content
                if isinstance(content, str)
                else str(content)
            ),
        })

    return result
