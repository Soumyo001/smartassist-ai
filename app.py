from datetime import date, datetime
from html import escape
import logging
from zoneinfo import ZoneInfo

import streamlit as st

from task_agent.agent import run_turn
from task_agent.chat_history import load_chat_history
from task_agent.storage import init_db, list_tasks
from task_agent.tool_policy import get_enabled_tool_names


THREAD_ID = "smartassist-web-demo"
TIMEZONE = ZoneInfo("Asia/Dhaka")


st.set_page_config(
    page_title="SmartAssist AI | Task Manager",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------
# DESIGN
# -------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #f4f7fc !important;
    color: #172640 !important;
}

/* Remove Streamlit's top header and toolbar */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer,
#MainMenu {
    display: none !important;
}

section[data-testid="stMain"] {
    background: #f4f7fc !important;
}

.block-container {
    max-width: 1390px !important;
    padding: 22px 26px 36px !important;
}

/* Typography */
[data-testid="stMarkdownContainer"] h1.page-heading {
    color: #172640 !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em !important;
    font-size: clamp(30px, 3vw, 40px) !important;
    line-height: 1.15 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #475c7b;
}

/* Top header */
.brandbar {
    background: #182b4d;
    color: #fff;
    border-radius: 18px;
    padding: 22px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 10px 25px rgba(21,43,79,.10);
    margin-bottom: 27px;
}

.brandbar strong {
    font-size: 24px;
    font-weight: 850;
    letter-spacing: .055em;
    color: #fff;
}

.brandbar span {
    font-size: 12px;
    color: #c2d3ee;
    letter-spacing: .12em;
    font-weight: 750;
}

.subtitle {
    color: #687e9d;
    margin: 5px 0 20px;
    font-size: 15px;
}

/* Dashboard statistics */
.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

.stats > div {
    background: #fff;
    border: 1px solid #e0e8f4;
    border-radius: 12px;
    padding: 13px 15px;
}

.stats small {
    display: block;
    color: #6d83a2;
    font-size: 12px;
}

.stats strong {
    display: block;
    color: #203a64;
    font-size: 23px;
    margin-top: 2px;
}

/* Task board */
.section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 13px;
}

.section-head strong {
    color: #1b2d4a;
    font-size: 20px;
}

.count-pill {
    color: #3c5da2;
    background: #eaf0fc;
    font-size: 13px;
    font-weight: 750;
    padding: 7px 13px;
    border-radius: 10px;
}

.task-card {
    display: flex;
    align-items: center;
    gap: 13px;
    border: 1px solid #dfe7f3;
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    background: #f8faff;
}

.task-dot {
    color: #789bd7;
    font-size: 23px;
    line-height: 1;
    flex-shrink: 0;
}

.task-dot.done {
    color: #208966;
}

.task-main {
    flex: 1;
    min-width: 0;
}

.task-name {
    font-size: 16px;
    color: #172640;
    font-weight: 750;
    overflow-wrap: anywhere;
}

.task-name.done {
    text-decoration: line-through;
    color: #8997ad;
}

.task-meta {
    color: #69809f;
    font-size: 12px;
    margin-top: 4px;
}

.task-description {
    color: #607799;
    font-size: 12px;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

/* Priority badges */
.priority-pill {
    border-radius: 9px;
    padding: 8px 12px;
    min-width: 74px;
    text-align: center;
    font-size: 12px;
    font-weight: 750;
}

.priority-high {
    background: #ffebe7;
    color: #b84332;
}

.priority-medium {
    background: #fff0d5;
    color: #936009;
}

.priority-low {
    background: #def2e9;
    color: #217555;
}

/* Assistant */
.assistant-top {
    background: #eaf0fd;
    border-radius: 13px;
    padding: 15px 17px;
    color: #1b3d70;
    font-size: 20px;
    font-weight: 800;
}

.assistant-sub {
    color: #7187a4;
    margin: 9px 0 14px;
    font-size: 13px;
}

div[data-testid="stChatMessage"] {
    background: #f5f8fe !important;
    border: 1px solid #e2eafa !important;
    border-radius: 15px !important;
    padding: 10px 12px !important;
    margin-bottom: 9px !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span {
    color: #1c3153 !important;
    overflow-wrap: anywhere;
}

div[data-testid="stChatInput"] {
    border-radius: 13px !important;
}

div[data-testid="stChatInput"] textarea {
    background: #fff !important;
    color: #1b2b44 !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #8799b2 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
}

/* Responsive layout */
@media(max-width: 768px) {
    .block-container {
        padding: 12px 13px 28px !important;
    }

    .brandbar {
        padding: 16px;
        margin-bottom: 18px;
    }

    .brandbar strong {
        font-size: 18px;
    }

    .brandbar span {
        font-size: 10px;
    }

    .task-card {
        padding: 12px;
    }

    .task-name {
        font-size: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def due_label(value: str | None) -> str:
    if not value:
        return "No due date"

    try:
        task_date = date.fromisoformat(value)
    except ValueError:
        return value

    today = datetime.now(TIMEZONE).date()

    if task_date == today:
        return "Today"

    if task_date == date.fromordinal(
        today.toordinal() + 1
    ):
        return "Tomorrow"

    return task_date.strftime("%d %b %Y")


def render_task(task: dict) -> None:
    title = escape(task["title"])
    description = escape(
        task.get("description") or ""
    )

    status = escape(task["status"])
    priority = task["priority"]

    priority_class = {
        "High": "priority-high",
        "Medium": "priority-medium",
        "Low": "priority-low",
    }.get(priority, "priority-medium")

    done = status == "Done"

    detail = (
        f'<div class="task-description">'
        f'{description}</div>'
        if description
        else ""
    )

    st.markdown(f"""
<div class="task-card">
  <span class="task-dot {'done' if done else ''}">
    {'✓' if done else '○'}
  </span>

  <div class="task-main">
    <div class="task-name {'done' if done else ''}">
      {title}
    </div>

    <div class="task-meta">
      #{int(task['id'])}
      · {escape(due_label(task.get('due_date')))}
      · {status}
    </div>

    {detail}
  </div>

  <span class="priority-pill {priority_class}">
    {escape(priority)}
  </span>
</div>
""", unsafe_allow_html=True)


def render_message(
    role: str,
    content: str
) -> None:

    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):

        if role == "user":
            st.markdown(
                f'<div style="color:#1c3153;'
                f'white-space:pre-wrap;'
                f'overflow-wrap:anywhere">'
                f'{escape(content)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)


# -------------------------------------------------
# INITIAL STATE
# -------------------------------------------------

init_db()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = (
        load_chat_history(THREAD_ID)
    )


# -------------------------------------------------
# BRAND HEADER
# -------------------------------------------------

st.markdown("""
<div class="brandbar">
  <strong>✦ SMARTASSIST</strong>
  <span>AI TASK MANAGER</span>
</div>
""", unsafe_allow_html=True)


left, right = st.columns(
    [1.60, 1],
    gap="large"
)


# -------------------------------------------------
# TASK BOARD
# -------------------------------------------------

with left:

    st.markdown(
        '<h1 class="page-heading">My tasks</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Organize work by asking your AI assistant.'
        '</div>',
        unsafe_allow_html=True,
    )

    tasks = list_tasks()

    done = sum(
        task["status"] == "Done"
        for task in tasks
    )

    ongoing = sum(
        task["status"] == "In progress"
        for task in tasks
    )

    st.markdown(f"""
<div class="stats">
  <div>
    <small>Total tasks</small>
    <strong>{len(tasks)}</strong>
  </div>
  <div>
    <small>In progress</small>
    <strong>{ongoing}</strong>
  </div>
  <div>
    <small>Completed</small>
    <strong>{done}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown(
            f'<div class="section-head">'
            f'<strong>Task board</strong>'
            f'<span class="count-pill">'
            f'All tasks {len(tasks)}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        if not tasks:
            st.info(
                "No tasks yet. Tell the AI assistant "
                "to create one."
            )

        for task in tasks:
            render_task(task)


# -------------------------------------------------
# AI CHAT
# -------------------------------------------------

with right:

    with st.container(border=True):

        st.markdown(
            '<div class="assistant-top">'
            '✦ &nbsp; AI Assistant</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="assistant-sub">'
            'Manage everything with a message'
            '</div>',
            unsafe_allow_html=True,
        )

        # Keeps the conversation above the input.
        chat_area = st.container(
            height=455,
            border=False,
        )

        with chat_area:

            if not st.session_state.chat_messages:
                st.info(
                    'Try: "Create a high-priority task '
                    'to finish the demo tomorrow."'
                )

            for item in st.session_state.chat_messages[-40:]:
                render_message(
                    item["role"],
                    item["content"],
                )

        # Native chat input: Enter sends.
        prompt = st.chat_input(
            "Type a task instruction and press Enter...",
            key="smartassist_chat_input",
            max_chars=2000,
        )

        st.caption(
            "Enter to send · Shift+Enter for a new line · "
            f"{len(get_enabled_tool_names())} approved tools"
        )

        # Process immediately after submission.
        if prompt and prompt.strip():

            prompt = prompt.strip()

            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt,
            })

            # Draw new messages INSIDE chat_area.
            # This appears before the LLM finishes.
            with chat_area:

                render_message("user", prompt)

                with st.chat_message(
                    "assistant",
                    avatar="🤖",
                ):

                    with st.status(
                        "SmartAssist AI is thinking...",
                        expanded=True,
                    ) as status:

                        st.write(
                            "Understanding your request "
                            "and checking approved tools..."
                        )

                        try:
                            reply = run_turn(
                                prompt,
                                thread_id=THREAD_ID,
                            )

                        except Exception:
                            logging.exception(
                                "SmartAssist request failed"
                            )

                            status.update(
                                label="Request failed",
                                state="error",
                                expanded=False,
                            )

                            error_text = (
                                "I couldn't complete that request. "
                                "Check the task board before retrying; "
                                "a previous operation might have succeeded."
                            )

                            st.error(error_text)

                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": error_text,
                            })

                        else:
                            status.update(
                                label="Request finished",
                                state="complete",
                                expanded=False,
                            )

                            st.markdown(reply)

                            # Reload saved conversation
                            # and refresh the task board.
                            st.session_state.chat_messages = (
                                load_chat_history(THREAD_ID)
                            )

                            st.rerun()
