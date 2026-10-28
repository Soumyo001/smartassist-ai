# SmartAssist-AI — AI Agent Task Manager

**A chat-first, single-page task manager powered by Groq, LangGraph, Streamlit, SQLite, and ChromaDB.**

SmartAssist-AI was developed for the **AI Agent Task Manager** internship assignment (AI & Vibe Coding). It lets users create, find, organize, update, complete, and delete tasks using natural-language instructions. The task board and AI conversation appear together, so users can see the results of their commands without switching pages.

**GitHub:** https://github.com/Soumyo001/smartassist-ai

## Features

- **Chat-controlled task management:** Create, list, view, edit, change status or priority, complete, and delete tasks.
- **Complete task records:** Stable numeric ID, title, optional description, status (`To do`, `In progress`, `Done`), priority (`Low`, `Medium`, `High`), and optional due date.
- **Semantic search:** Find related tasks through natural-language queries, using ChromaDB for vector retrieval.
- **Persistent data:** SQLite stores task records across page refreshes and application restarts.
- **Structured execution:** Groq generates validated action plans; LangGraph executes approved operations in sequence.
- **Safer modifications:** Ambiguous updates/deletions require clarification. Unsupported task merging and bulk deletion are blocked.
- **Task planning assistance:** Request a specified number of recommended unfinished tasks based on urgency and priority, without changing any records.
- **Conversation continuity:** LangGraph checkpoints preserve chat state and pending clarifications.
- **Responsive single-page interface:** The task board, conversation area, instruction field, and send control share one Streamlit page.

## Tech stack

| Component | Technology | Purpose |
| --- | --- | --- |
| User interface | Streamlit | Single-page task board and chat |
| AI model | GroqCloud-hosted GPT-OSS | Natural-language understanding and structured planning |
| Agent orchestration | LangGraph + LangChain | Controlled execution, state, and conversation workflow |
| Task database | SQLite | Authoritative task records and stable integer IDs |
| Vector database | ChromaDB | Embeddings and semantic task retrieval |
| Validation | Pydantic | Task fields and structured agent actions |
| Testing | pytest | Automated regression and behavior tests |

### Why Groq?

GroqCloud provides hosted LLM inference and structured-output support, without requiring a locally hosted language model. The model is configurable through `GROQ_MODEL` in `.env`; the project's default is `openai/gpt-oss-20b`, and `openai/gpt-oss-120b` can also be selected if available to the account.

Groq is responsible for **interpreting requests and producing structured plans**. The application—not the language model—validates tool access, resolves actual task IDs, performs database operations, and formats many results. Straightforward requests may take a deterministic route without an LLM call. Task recommendations use Groq to identify advice intent and a rule-based ranking function to select tasks.

## How it works

1. The user sends a task instruction through the Streamlit chat interface.
2. The application interprets it using deterministic routing or a Groq-generated structured `ActionPlan`.
3. Pydantic checks the action schema, and the tool permission list restricts executable operations.
4. The agent resolves task references against SQLite. If a modification or deletion targets multiple possible tasks, it requests clarification instead of guessing.
5. LangGraph executes authorized actions sequentially and returns task-specific success or error messages.
6. SQLite remains the **single source of truth**. The task board reflects saved data after updates and reloads.
7. For semantic requests, ChromaDB retrieves similar task candidates from indexed titles/descriptions; current task details are verified against SQLite.
8. LangGraph's SQLite checkpoint store maintains conversation context between turns.

The seven available tools are `create_task`, `list_tasks`, `get_task`, `update_task`, `complete_task`, `delete_task`, and `search_tasks`. They are controlled through `config/tool_permissions.json`.

## Getting started

### Prerequisites

- Ubuntu/Linux with **Python 3.12** (the development environment)
- Internet access for Groq API requests
- A **Groq API key**
- Internet access for the initial download of ChromaDB's default embedding model, if not already cached

### 1. Clone and install

```bash
git clone https://github.com/Soumyo001/smartassist-ai.git
cd smartassist-ai

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and provide your own Groq API key. For example:

```dotenv
GROQ_API_KEY=YOUR_PRIVATE_GROQ_API_KEY
GROQ_MODEL=openai/gpt-oss-20b
GROQ_REASONING_EFFORT=low
USER_TIMEZONE=Asia/Dhaka
```

`GROQ_MODEL` can be changed to `openai/gpt-oss-120b` if your Groq account supports it. The Streamlit interface also provides a reasoning-effort selector for individual requests.

**Security:** Never commit or upload `.env` or API credentials. The provided `.gitignore` excludes `.env`, the virtual environment, and runtime data.

### 3. Run SmartAssist-AI

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The task board is displayed next to the AI chat interface on desktop; the layout adapts to narrower screens.

For a terminal-only alternative:

```bash
python chat_cli.py
```

### 4. Run tests

```bash
python -m pytest -q
```

**Latest development result:** `73 passed` on October 28, 2026. This is the result of the developer's local test run, not a guarantee that every external API call will succeed.

Optionally, verify live Groq structured output:

```bash
python verify_groq.py
```

This optional verification **calls the real Groq API**, requires a valid key, and counts toward the account's API usage. The regular pytest suite primarily uses isolated fixtures and simulated planners rather than relying on live Groq requests.

## Five demonstration prompts

Use these prompts to demonstrate the assignment's core requirements. For operations involving existing tasks, make sure the named task exists on the board; replace ambiguous names with actual task IDs when necessary.

| # | Prompt | What it demonstrates |
| --- | --- | --- |
| 1 | `Create a high-priority task to finish the demo tomorrow.` | Natural-language task creation, priority, and due date |
| 2 | `Show all unfinished tasks related to documentation.` | Semantic search and status filtering |
| 3 | `Move the demo task to In progress and mark the weekly report Done.` | Two sequential operations, status changes, and task resolution |
| 4 | `Change the priority of the API task to Low.` | Editing a task through conversational instructions |
| 5 | `Which five tasks should I focus on first, and why?` | Read-only recommendations based on deadlines and priorities |

**Deletion and clarification demonstration:** You can also ask the agent to delete a **single test task** by its actual ID. For ambiguous names, the agent asks you to select the intended task. Multiple-task deletion requests are intentionally blocked. Avoid deleting important records during a live demo.

**Persistence demonstration:** Create or update a test task, refresh the browser, then restart Streamlit. The task should retain its saved ID and values because SQLite stores the task state.

## Validation, safety, and reliability

- **Strict schemas:** Pydantic validates task fields and structured plans before execution.
- **Permission controls:** The approved action allowlist prevents unauthorized tools from being invoked.
- **No invented IDs:** Model-generated IDs are not trusted unless supported by the user's explicit task reference.
- **Ambiguity handling:** Potentially ambiguous modifications and deletions require user selection.
- **Deletion safeguards:** Unsupported merges do not trigger deletion; requests to delete multiple tasks are rejected.
- **Date validation:** Invalid dates are rejected instead of silently altering deadlines. Supported follow-ups can correct a previously rejected date for the intended task.
- **Search relevance:** A calibrated similarity-distance cutoff filters weak ChromaDB results. Retrieved records are checked against SQLite.
- **Task-data safety:** Saved titles and descriptions are treated as data, not instructions to the agent.
- **Clear outcomes:** The interface shows task-specific results and error messages; the backend does not silently treat failed actions as successful.

## Problems encountered and how they were fixed

These are actual development issues investigated through live conversation tests and automated regression tests.

| Issue encountered | Diagnosis and correction | Verification |
| --- | --- | --- |
| **Wrong task list for explicit IDs** | A request to show two specific tasks was treated as a general list. Added a narrow read-only route that retrieves only the supplied IDs. | Confirmed that `Show task #20 and task #21 separately` returned exactly two records. |
| **Compound sorting failed** | Priority order was applied, but due dates were not sorted within each priority group. Added stable secondary date sorting, supporting latest/descending and urgent/ascending dates. | Regression tests and live queries verified both directions. |
| **Ambiguous or unsafe deletion** | A model could infer an unintended target from a partial name or merge request. Added explicit authorization checks, clarification, and bulk-delete blocking. | Safety tests verify that unapproved deletions do not change task records. |
| **Invalid date and lost follow-up context** | An invalid due-date value could cause a misleading action or disrupt follow-up editing. Added strict date validation and safe contextual date correction. | Tests verify rejection of invalid values and correct updates to the intended record. |
| **Irrelevant semantic matches** | Nearest vector neighbors were not always relevant. Added a ChromaDB similarity-distance threshold and SQLite verification. | Regression tests reject weak matches while retaining relevant task results. |
| **Recommendation count and priority** | Asking for five suggestions returned three; a distant deadline also outranked higher-priority work. Added requested-count extraction and a near-term urgency policy. | Live five-task recommendations and ranking/count regression tests passed. |
| **Groq rate limit** | A Groq request returned HTTP 429 because the account's tokens-per-minute allowance was exceeded. | Identified as an external API quota issue; automatic rate-limit recovery has **not** been implemented. |

## Known limitations

- **Hosted API dependency:** Groq requests require network access, a valid API key, and available provider quota. HTTP 429 can cause a request to fail; automatic rate-limit recovery is not currently implemented.
- **Approximate semantic retrieval:** Vector similarity can miss relevant phrasing or surface weak matches; the configured relevance threshold is calibrated for the current project dataset.
- **No reminders or notifications:** Due dates are stored as calendar dates. The app does not send background reminders or alarms.
- **Task advice is rule-based after intent detection:** Recommendations use stored status, priority, and deadline information rather than a separate autonomous planning service.
- **Local application scope:** This is an internship demonstration project, not a production multi-user task service with authentication and hosted synchronization.

## Project structure

```text
smartassist-ai/
├── app.py                         # Single-page Streamlit interface
├── chat_cli.py                    # Optional terminal chat
├── task_agent/
│   ├── agent.py                   # Groq planning and LangGraph workflow
│   ├── tools.py                   # Task tool definitions
│   ├── tool_policy.py             # Approved tool enforcement
│   ├── storage.py                 # SQLite task operations
│   ├── schemas.py                 # Pydantic task models
│   ├── vector_store.py            # ChromaDB indexing and semantic search
│   ├── response_formatting.py     # Task formatting, sorting, recommendations
│   ├── task_intelligence.py       # Task resolution and safety helpers
│   └── chat_history.py            # Persistent conversation loading
├── config/
│   └── tool_permissions.json      # Enabled task operations
├── tests/                         # Automated tests and regressions
├── verify_groq.py                 # Optional live API verification
├── verify_step1.py                # Initial setup checks
├── requirements.txt
├── .env.example
└── README.md
```

SQLite task records, conversation checkpoints, and the local ChromaDB index are generated at runtime under `data/` and are intentionally not committed to GitHub.

## Assignment coverage and submission

This repository addresses the assignment's core requirements: **chat-first single-page UI, complete task CRUD, task status and priority control, semantic search with ChromaDB, persistent SQLite task state, structured actions, validation, and clarification for ambiguous task changes.**

The assignment also requires a **short demo video or live demonstration** alongside the repository link and README. Use the five prompts above and demonstrate persistence, semantic retrieval, and a safe task update during the presentation.

**Repository:** https://github.com/Soumyo001/smartassist-ai

## License

MIT — see [`LICENSE`](LICENSE).