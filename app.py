from datetime import date, datetime, timedelta
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


# ----------------------------------------
# DESIGN
# ----------------------------------------

st.html("""
<style>
.stApp, section[data-testid="stMain"] {
    background: #f4f7fd !important;
}

header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer {
    display: none !important;
}

.block-container {
    max-width: 1350px !important;
    padding: 25px 25px 45px !important;
}

/* Brand header */
.brandbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: linear-gradient(110deg,#152a4d,#203d6a);
    color: #fff;
    border-radius: 18px;
    padding: 22px 27px;
    margin-bottom: 25px;
    box-shadow: 0 15px 32px rgba(21,42,77,.13);
    position: relative;
    z-index: 2;
    isolation: isolate;
    overflow: hidden;
    border: 0 !important;
}

.brand {
    font-size: 22px;
    font-weight: 850;
    letter-spacing: .045em;
    color: #fff;
}

.brand-note {
    font-size: 11px;
    letter-spacing: .13em;
    font-weight: 800;
    color: #bed3f5;
}

/* Page titles */
.page-heading {
    font-size: 33px;
    font-weight: 850;
    letter-spacing: -.04em;
    color: #172a49;
}

.page-subtitle {
    color: #7385a2;
    font-size: 14px;
    margin: 4px 0 19px;
}

/* Dashboard metrics */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3,minmax(0,1fr));
    gap: 10px;
    margin: 18px 0 17px;
}

.metric {
    background: #fff;
    border: 1px solid #dde7f4;
    border-radius: 13px;
    padding: 15px 17px;
}

.metric-label {
    display: block;
    color: #7587a2;
    font-size: 12px;
}

.metric-number {
    display: block;
    color: #172d52;
    font-weight: 800;
    font-size: 27px;
    margin-top: 3px;
}

/* Task board */
.section-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 13px;
    color: #203451;
    font-weight: 800;
    font-size: 18px;
}

.count-pill {
    background: #e8efff;
    color: #315ba3;
    border-radius: 9px;
    padding: 7px 12px;
    font-size: 12px;
}

.task-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.task-card {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 13px 15px;
    background: #f9fbff;
    border: 1px solid #dde7f4;
    border-radius: 12px;
}

.task-status-icon {
    flex: none;
    font-size: 22px;
    color: #829ed1;
    line-height: 1;
}

.task-status-icon.is-done {
    color: #268563;
}

.task-main {
    min-width: 0;
    flex: 1;
}

.task-name {
    color: #182c4d;
    font-weight: 750;
    font-size: 15px;
    overflow-wrap: anywhere;
}

.task-name.is-done {
    text-decoration: line-through;
    color: #8999b1;
}

.task-meta {
    font-size: 12px;
    color: #7b8eab;
    margin-top: 5px;
}

.task-description {
    font-size: 12px;
    color: #677c9c;
    margin-top: 5px;
    overflow-wrap: anywhere;
}

/* Priority badges */
.priority-pill {
    flex: none;
    min-width: 70px;
    text-align: center;
    border-radius: 9px;
    padding: 8px 11px;
    font-size: 11px;
    font-weight: 750;
}

.priority-high {
    background: #ffe8e5;
    color: #b74237;
}

.priority-medium {
    background: #fff1d5;
    color: #98600b;
}

.priority-low {
    background: #dff3e9;
    color: #247756;
}

/* Assistant */
.assistant-top {
    background: #e8effc;
    color: #1c3e70;
    border-radius: 13px;
    padding: 15px 16px;
    font-weight: 850;
    font-size: 19px;
}

.assistant-sub {
    font-size: 12px;
    color: #7a8aa6;
    margin: 10px 0 14px;
}

/* Chat layout and avatar alignment */
div[data-testid="stChatMessage"] {
    align-items: center !important;
    border: 1px solid #e0e8f7;
    background: #f8faff !important;
    border-radius: 13px !important;
    padding: 12px !important;
    margin: 7px 0 !important;
}

div[data-testid="stChatMessage"]
[data-testid="stChatMessageAvatarUser"],
div[data-testid="stChatMessage"]
[data-testid="stChatMessageAvatarAssistant"] {
    align-self: center !important;
}

div[data-testid="stChatMessageContent"] {
    min-width: 0 !important;
    align-self: center !important;
}

div[data-testid="stChatMessageContent"] p {
    margin: 0.18rem 0 !important;
    color: #1b3152 !important;
    line-height: 1.55;
    overflow-wrap: anywhere;
}

div[data-testid="stChatMessageContent"] ul,
div[data-testid="stChatMessageContent"] ol {
    padding-left: 1.1rem;
    margin: .4rem 0 !important;
}

.user-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    color: #183154;
    line-height: 1.45;
    margin: 0;
}

div[data-testid="stChatInput"] textarea {
    background: #fff !important;
    color: #172d50 !important;
}

div[data-testid="stChatInput"] {
    border-radius: 13px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 17px !important;
}

/* Responsive adjustments: keep chat easy to reach on phones and tablets. */
@media(max-width:900px) {
    /* This app has one st.columns() row: board (first) + chat (second).
       Stack it and place the primary chat control before the long board. */
    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 18px !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        flex: 1 1 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
        order: 2;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
        order: 1;
    }

    .block-container {
        padding: 14px 13px 30px !important;
    }

    .brandbar {
        padding: 17px;
        margin-bottom: 18px;
    }

    .brand {
        font-size: 18px;
    }

    .brand-note {
        font-size: 9px;
    }

    .task-card {
        padding: 12px;
    }

    .task-name {
        font-size: 14px;
    }

    .metric {
        padding: 11px;
    }

    .metric-number {
        font-size: 21px;
    }
}
</style>
""")


# ----------------------------------------
# HELPERS
# ----------------------------------------

def due_label(value: str | None) -> str:
    if not value:
        return "No due date"

    try:
        due = date.fromisoformat(value)
    except ValueError:
        return value

    today = datetime.now(TIMEZONE).date()

    if due == today:
        return "Today"

    if due == today + timedelta(days=1):
        return "Tomorrow"

    return due.strftime("%d %b %Y")


def task_card_html(task: dict) -> str:
    """
    Generate safe HTML for a task card.
    Escape all user-created task content.
    """

    done = task["status"] == "Done"

    pill = {
        "High": "priority-high",
        "Medium": "priority-medium",
        "Low": "priority-low",
    }.get(task["priority"], "priority-medium")

    title = escape(task["title"])

    meta = escape(
        f"#{task['id']} · "
        f"{due_label(task.get('due_date'))} · "
        f"{task['status']}"
    )

    description = escape(
        task.get("description") or ""
    )

    description_html = (
        f'<div class="task-description">'
        f'{description}</div>'
        if description
        else ""
    )

    done_class = "is-done" if done else ""
    icon = "✓" if done else "○"

    return (
        '<div class="task-card">'
        f'<span class="task-status-icon {done_class}">{icon}</span>'
        '<div class="task-main">'
        f'<div class="task-name {done_class}">{title}</div>'
        f'<div class="task-meta">{meta}</div>'
        f'{description_html}'
        '</div>'
        f'<span class="priority-pill {pill}">'
        f'{escape(task["priority"])}</span>'
        '</div>'
    )


def render_message(role: str, content: str) -> None:

    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):

        if role == "user":
            st.html(
                f'<p class="user-text">'
                f'{escape(content)}</p>'
            )
        else:
            st.markdown(content)


# ----------------------------------------
# INITIALIZATION
# ----------------------------------------

init_db()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = (
        load_chat_history(THREAD_ID)
    )


# ----------------------------------------
# BRAND HEADER
# ----------------------------------------

st.html(
    '<div class="brandbar">'
    '<div class="brand">✦ SMARTASSIST</div>'
    '<div class="brand-note">AI TASK MANAGER</div>'
    '</div>'
)


left, right = st.columns(
    [1.55, 1],
    gap="large"
)


# ----------------------------------------
# LEFT — TASK BOARD
# ----------------------------------------

with left:

    st.html(
        '<div class="page-heading">My tasks</div>'
        '<div class="page-subtitle">'
        'Organize your day with SmartAssist AI.'
        '</div>'
    )

    tasks = list_tasks()

    ongoing = sum(
        t["status"] == "In progress"
        for t in tasks
    )

    done = sum(
        t["status"] == "Done"
        for t in tasks
    )

    st.html(
        '<div class="metric-strip">'
        f'<div class="metric">'
        f'<span class="metric-label">Total tasks</span>'
        f'<span class="metric-number">{len(tasks)}</span>'
        '</div>'
        f'<div class="metric">'
        f'<span class="metric-label">In progress</span>'
        f'<span class="metric-number">{ongoing}</span>'
        '</div>'
        f'<div class="metric">'
        f'<span class="metric-label">Completed</span>'
        f'<span class="metric-number">{done}</span>'
        '</div>'
        '</div>'
    )

    with st.container(border=True):

        st.html(
            '<div class="section-heading">'
            '<span>Task board</span>'
            f'<span class="count-pill">'
            f'All tasks {len(tasks)}'
            '</span></div>'
        )

        if tasks:
            st.html(
                '<div class="task-list">'
                + "".join(
                    task_card_html(t)
                    for t in tasks
                )
                + '</div>'
            )
        else:
            st.info(
                "No tasks yet. Ask SmartAssist to create one!"
            )


# ----------------------------------------
# RIGHT — AI CHAT
# ----------------------------------------

with right:

    with st.container(border=True):

        st.html(
            '<div class="assistant-top">'
            '✦ &nbsp; AI Assistant'
            '</div>'
            '<div class="assistant-sub">'
            'Manage everything with a message'
            '</div>'
        )

        reasoning_mode = st.segmented_control(
            "Reasoning effort",
            options=["Low", "Medium", "High"],
            selection_mode="single",
            default="Medium",
            required=True,
            key="smartassist_reasoning_mode",
            help=(
                "Low: simple requests. "
                "Medium: balanced. "
                "High: complex task planning."
            ),
        )

        chat_area = st.container(
            height=400,
            border=False,
        )

        with chat_area:

            if not st.session_state.chat_messages:
                st.info(
                    'Try: "Create a high-priority task '
                    'for tomorrow."'
                )

            for item in st.session_state.chat_messages[-40:]:
                render_message(
                    item["role"],
                    item["content"],
                )
        
        last_activity = st.session_state.get(
            "last_agent_activity"
        )

        if last_activity:
            with st.expander(
                "Last request · execution activity",
                expanded=False,
            ):
                st.caption(
                    "Reasoning setting: "
                    + last_activity["effort"]
                )

                for step in last_activity["steps"]:
                    st.markdown(f"- {step}")

        prompt = st.chat_input(
            "Type a task instruction and press Enter...",
            key="smartassist_chat_input",
            max_chars=2000,
        )

        st.caption(
            "Enter to send · Shift+Enter for a new line · "
            f"{len(get_enabled_tool_names())} approved tools"
        )

        if prompt and prompt.strip():

            prompt = prompt.strip()

            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt,
            })

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

                        try:
                            progress_line = st.empty()
                            activity = []

                            def report_progress(stage: str) -> None:
                                if not activity or activity[-1] != stage:
                                    activity.append(stage)

                                progress_line.markdown(
                                    "**Execution activity**\n\n"
                                    + "\n".join(
                                        f"- {step}"
                                        for step in activity
                                    )
                                )

                            response = run_turn(
                                prompt,
                                thread_id=THREAD_ID,
                                reasoning_effort=reasoning_mode.lower(),
                                on_progress=report_progress,
                            )

                        except Exception:

                            logging.exception(
                                "SmartAssist request failed"
                            )

                            status.update(
                                label="Request failed",
                                state="error",
                            )

                            message = (
                                "I couldn't finish that request. "
                                "Check the task board before retrying; "
                                "an earlier action may have succeeded."
                            )

                            st.error(message)

                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": message,
                            })

                        else:

                            status.update(
                                label="Done",
                                state="complete",
                                expanded=False,
                            )

                            st.markdown(response)

                            st.session_state.chat_messages = (
                                load_chat_history(THREAD_ID)
                            )

                            st.session_state["last_agent_activity"] = {
                                "effort": reasoning_mode,
                                "steps": activity.copy(),
                            }

                            st.rerun()
