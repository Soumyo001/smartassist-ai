from task_agent.agent import run_turn


def main():

    print("=" * 45)
    print("SMARTASSIST AI — TASK MANAGER")
    print("=" * 45)

    print("Type 'exit' to quit.")
    print("Type 'cancel' to cancel pending clarification.")

    while True:

        try:
            instruction = input("\nYou: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if instruction.lower() in {
            "exit", "quit"
        }:
            print("Goodbye!")
            break

        if not instruction:
            continue

        print("\nSmartAssist AI is thinking...")

        try:
            response = run_turn(
                instruction,
                thread_id="default-user"
            )

            print(f"\nSmartAssist AI:\n{response}")

        except Exception as error:
            print(
                f"\nError: {type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()