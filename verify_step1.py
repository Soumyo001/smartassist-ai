import argparse
import os
import sqlite3
import tempfile

from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from task_agent.storage import (
    init_db,
    DB_PATH,
    list_tasks
)
from task_agent.vector_store import get_task_collection


load_dotenv()


class AgentTestState(TypedDict):
    response: str


def check_sqlite():
    init_db()

    print(f"PASS SQLite ready: {DB_PATH}")
    print(f"PASS Existing tasks: {len(list_tasks())}")

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "test.sqlite3"

        init_db(database)

        with sqlite3.connect(database) as conn:
            conn.execute(
                "INSERT INTO tasks (title) VALUES (?)",
                ("Persistence test",)
            )
            conn.commit()

        assert len(list_tasks(database)) == 1

        print("PASS SQLite persistence")


def check_chroma():
    collection = get_task_collection()

    assert collection.name == "smartassist_tasks"

    print(
        f"PASS Chroma collection: {collection.name}"
    )


def check_llm():
    from langchain_openrouter import ChatOpenRouter
    from langgraph.graph import StateGraph, START, END

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or api_key.startswith("YOUR_"):
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. "
            "Add your new OpenRouter key to .env."
        )

    model_name = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    )

    model = ChatOpenRouter(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=60,
        max_retries=0
    )

    def agent_node(state: AgentTestState):
        answer = model.invoke(
            "Reply with exactly this sentence: "
            "SmartAssist AI model connected."
        )

        return {
            "response": str(answer.content)
        }

    graph = StateGraph(AgentTestState)

    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    agent = graph.compile()

    result = agent.invoke({
        "response": ""
    })

    assert result["response"].strip(), (
        "The model returned an empty response."
    )

    print(f"PASS OpenRouter model: {model_name}")
    print(f"PASS LangGraph response: {result['response']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llm",
        action="store_true",
        help="Test OpenRouter (may use paid API credits)"
    )

    args = parser.parse_args()

    print("=" * 45)
    print("SmartAssist AI - Step 1 Verification")
    print("=" * 45)

    check_sqlite()
    check_chroma()

    if args.llm:
        check_llm()
    else:
        print("SKIP OpenRouter API test")

    print("=" * 45)
    print("STEP 1 VERIFICATION PASSED")
    print("=" * 45)


if __name__ == "__main__":
    main()
