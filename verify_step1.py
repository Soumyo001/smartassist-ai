
import argparse
import os
import sqlite3
import tempfile

from pathlib import Path
from dotenv import load_dotenv

from task_agent.storage import (
    init_db,
    DB_PATH,
    list_tasks
)
from task_agent.vector_store import get_task_collection

load_dotenv()


def run():
    init_db()

    print(f"PASS SQLite ready: {DB_PATH}")
    print(f"PASS Existing tasks: {len(list_tasks())}")

    collection = get_task_collection()

    print(
        f"PASS Chroma collection: {collection.name}"
    )

    with tempfile.TemporaryDirectory() as directory:

        database = Path(directory) / "persist.sqlite3"
        init_db(database)

        with sqlite3.connect(database) as conn:
            conn.execute(
                "INSERT INTO tasks (title) VALUES (?)",
                ("Persistence test",)
            )
            conn.commit()

        assert len(list_tasks(database)) == 1

        print("PASS SQLite persistence")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llm",
        action="store_true",
        help="Make one paid OpenRouter API request"
    )

    args = parser.parse_args()

    if args.llm:
        from langchain_openrouter import ChatOpenRouter
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        if not os.getenv("OPENROUTER_API_KEY"):
            raise SystemExit(
                "Add OPENROUTER_API_KEY to .env"
            )

        model = ChatOpenRouter(
            model=os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-4o-mini"
            ),
            temperature=0,
            max_tokens=60
        )

        class State(TypedDict):
            response: str

        def agent_node(state: State):
            answer = model.invoke(
                "Respond with exactly: "
                "TaskFlow model connected"
            )
            return {
                "response": str(answer.content)
            }

        graph = StateGraph(State)
        graph.add_node("model", agent_node)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)

        result = graph.compile().invoke(
            {"response": ""}
        )

        assert result["response"].strip()

        print(
            "PASS LangGraph + OpenRouter:",
            result["response"][:150]
        )

    else:
        print(
            "SKIP paid OpenRouter request"
        )

    print("STEP 1 FOUNDATION PASSED")


if __name__ == "__main__":
    run()
