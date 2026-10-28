# SmartAssist-AI — AI Task Manager

A chat-first, single-page task manager built for the **AI Agent Task Manager** internship assignment. Users can manage tasks through natural-language instructions alongside a live task board.

**Repository:** https://github.com/Soumyo001/smartassist-ai  
**Development source of truth:** The user's latest local project in `~/Documents/smartassist-ai` may be newer than the remote repository.

## Implemented architecture

- **Interface:** Streamlit (`app.py`), with a task board and AI conversation on one page.
- **Agent:** LangGraph and LangChain; the planner uses GroqCloud's hosted `openai/gpt-oss-20b` model for structured task plans.
- **Actions:** Seven permission-controlled tools: `create_task`, `list_tasks`, `get_task`, `update_task`, `complete_task`, `delete_task`, and `search_tasks`.
- **Task persistence:** SQLite (`data/smartassist.sqlite3`) is the authoritative record of task state and IDs.
- **Semantic retrieval:** ChromaDB indexes task titles/descriptions and returns candidates; SQLite provides the current task details.
- **Agent conversation memory:** LangGraph SQLite checkpoints (`data/agent_checkpoints.sqlite3`).
- **Validation and safety:** Pydantic task schemas, server-side tool permissions, task-ID resolution and clarification of ambiguous mutations.

## Requirements

- Ubuntu/Linux (project developed with Python 3.12)
- A Groq API key and network access for hosted-model requests
- Python packages listed in `requirements.txt`
- Internet access may be needed on first use to download ChromaDB's default embedding model

## Local setup

```bash
cd ~/Documents/smartassist-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally and replace `YOUR_GROQ_API_KEY` with your **private** Groq key. Do not commit or share `.env`.

Start the app:

```bash
streamlit run app.py
```

Alternatively, run a terminal chat with `python chat_cli.py`.

## Run verification

```bash
source .venv/bin/activate
python -m pytest -q
python verify_groq.py
```

The unit tests use temporary files where designed to do so. `verify_groq.py` calls the live Groq provider; it requires a valid API key and may consume API usage. Check the test output on your own machine before claiming all tests pass.

## Demonstration prompts

1. `Create a high-priority task to finish the demo tomorrow.`
2. `Show all unfinished tasks related to documentation.`
3. `Move the demo task to In progress and mark the weekly report Done.`
4. `Change the priority of the API task to Low.`
5. `Delete the documentation task.` — if multiple tasks match, choose an explicit ID after the agent asks for clarification.

These are suggested demonstration inputs, not proof of a successful live execution.

## Notes on task behavior

Task records have stable integer IDs, a title, optional description and due date, and status/priority values. The backend validates task updates rather than trusting model-generated instructions. Deleting ambiguous tasks requires clarification. Unsupported task merging must not cause deletion. Due dates are stored as calendar dates, **not** scheduled notifications.

## Debugging history and limitations

During development, the project encountered Groq structured-output errors, task-priority audit ambiguity, an unsafe model-inferred merge/delete, and follow-up date context loss. The source contains specific safeguards and regression tests for these cases, including `tests/test_final_context_fixes.py`. The final context fixes were recently implemented; they should be re-tested against the latest local build before describing them as verified. Hosted-model responses can still fail, and semantic search is approximate.

## Submission checklist

- [ ] Confirm mandatory requirements against the instructor PDF using the current code and live behavior.
- [ ] Run automated tests and verify restart persistence and semantic retrieval.
- [ ] Verify responsive layout and visible chat send control in a browser.
- [ ] Push the latest reviewed local source to the repository (do not commit secrets or runtime data).
- [ ] Record a short demo video **or** prepare a live demo.
- [ ] Include 3–5 meaningful successful demonstration prompts and factual examples of bugs and their fixes.