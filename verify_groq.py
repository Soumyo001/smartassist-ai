
"""Verify Groq structured output without modifying tasks."""

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from task_agent.agent import (
    ActionPlan,
    build_groq_planner,
)


def main():

    print("=" * 45)
    print("SmartAssist AI — Groq Verification")
    print("=" * 45)

    planner = build_groq_planner()

    response = planner.invoke([
        SystemMessage(
            content=(
                "You are an AI task manager. "
                "Convert the user's instruction into "
                "a valid structured ActionPlan. "
                "Do not execute any operations."
            )
        ),
        HumanMessage(
            content=(
                "Create a high-priority task "
                "called Review slides."
            )
        ),
    ])

    assert isinstance(response, ActionPlan)

    assert len(response.operations) == 1

    operation = response.operations[0]

    assert operation.action == "create_task"
    assert operation.priority == "High"

    print("PASS Groq connection")
    print("PASS Pydantic structured output")

    print("\nParsed operation:")
    print(operation.model_dump_json(indent=2))

    print("\nGROQ VERIFICATION PASSED")


if __name__ == "__main__":
    main()
