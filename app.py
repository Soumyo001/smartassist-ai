import streamlit as st

from task_agent.storage import init_db, list_tasks


st.set_page_config(
    page_title="SmartAssist AI",
    page_icon="✅",
    layout="wide"
)

init_db()

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
}

.taskflow-header {
    background: #192b4c;
    color: white;
    padding: 20px 26px;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.taskflow-header strong {
    font-size: 24px;
    letter-spacing: .8px;
}

.taskflow-header span {
    font-size: 14px;
    color: #c9d8f0;
}
</style>

<div class="taskflow-header">
    <strong>Smartassist-AI</strong>
    <span>AI TASK MANAGER</span>
</div>
""", unsafe_allow_html=True)


board, chat = st.columns(
    [1.65, 1],
    gap="large"
)


with board:

    st.title("My tasks")

    st.caption(
        "Organize tasks by asking your AI assistant"
    )

    tasks = list_tasks()

    st.subheader(
        f"Task board · All tasks {len(tasks)}"
    )

    if not tasks:
        st.info(
            "No tasks yet. Task creation "
            "will be added in Step 2."
        )

    for task in tasks:

        st.markdown(
            f"**#{task['id']} · {task['title']}**"
        )

        st.caption(
            f"{task['status']} · "
            f"{task['priority']} · "
            f"{task['due_date'] or 'No due date'}"
        )

        st.divider()


with chat:

    st.subheader("✦ AI Assistant")

    st.caption(
        "Manage everything with a message"
    )

    st.info(
        "Chat actions will be connected "
        "in Step 4."
    )

    with st.form("chat_shell"):

        st.text_input(
            "YOUR INSTRUCTION",
            placeholder="Type a task instruction...",
            disabled=True
        )

        st.form_submit_button(
            "Send",
            disabled=True
        )