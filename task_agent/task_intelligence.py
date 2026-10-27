"""Deterministic task identification and clarification helpers."""

import re


def normalized_title(value: str) -> str:
    return " ".join(
        re.findall(r"\w+", value.casefold())
    )


def mentioned_task_ids(text: str) -> set[int]:
    """
    Recognize explicit database IDs:
    #8, $8, task 8, task #8.
    """
    return {
        int(match.group(1))
        for match in re.finditer(
            r"(?<!\w)(?:[#\$]\s*|"
            r"\btask\s*(?:id\s*)?[#\$]?\s*)"
            r"(\d+)(?!\d)",
            text,
            flags=re.I,
        )
    }


def selected_candidate(
    text: str,
    candidates: list[dict]
) -> int | None:
    """
    During clarification:
      '#8 one' = actual database ID 8
      '1'      = first displayed option
      'second' = second displayed option

    Only pure selections can resume a pending operation.
    """

    simple = " ".join(
        text.casefold().strip().split()
    )

    valid_ids = {
        int(task["id"])
        for task in candidates
    }

    explicit = re.fullmatch(
        r"(?:the\s*)?(?:task\s*)?"
        r"[#\$]\s*(\d+)"
        r"(?:\s*(?:one|task))?\s*[.!]?",
        simple,
    )

    if explicit:
        selected = int(explicit.group(1))

        return (
            selected
            if selected in valid_ids
            else None
        )

    label = re.fullmatch(
        r"task\s+(\d+)"
        r"(?:\s+one)?\s*[.!]?",
        simple,
    )

    if label:
        selected = int(label.group(1))

        return (
            selected
            if selected in valid_ids
            else None
        )

    ordinal = re.fullmatch(
        r"(?:the\s*)?"
        r"(first|second|third|fourth|fifth|\d+)"
        r"(?:\s+(?:one|option|task))?"
        r"\s*[.!]?",
        simple,
    )

    if ordinal:
        word = ordinal.group(1)

        positions = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
        }

        position = positions.get(word)

        if position is None:
            position = int(word)

        if 1 <= position <= len(candidates):
            return int(
                candidates[position - 1]["id"]
            )

    return None


def numbered_title(
    title: str,
    existing: list[dict]
) -> str:
    """
    Duplicate task titles become:
    Submit assignment
    Submit assignment (2)
    Submit assignment (3)
    """

    title = title.strip()

    root = re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        title,
    ).strip()

    if not root:
        return title

    used = {
        normalized_title(task["title"])
        for task in existing
    }

    if normalized_title(title) not in used:
        return title

    number = 2

    while normalized_title(
        f"{root} ({number})"
    ) in used:
        number += 1

    return f"{root} ({number})"


def is_placeholder_title(title: str) -> bool:
    return normalized_title(title) in {
        "task",
        "a task",
        "new task",
        "create task",
        "create a task",
        "create a new task",
        "untitled",
        "untitled task",
        "a new task",
    }


def is_vague_create_request(text: str) -> bool:
    words = " ".join(
        re.findall(r"\w+", text.casefold())
    )

    return bool(
        re.fullmatch(
            r"(?:(?:can|could|would) you |"
            r"please |i (?:want|need) to )?"
            r"(?:create|add|make)"
            r"(?: me)?"
            r"(?: a| the)?"
            r"(?: new)? task"
            r"(?: now| please| for me)?",
            words,
        )
    )


def is_plain_list_request(text: str) -> bool:
    """
    Simple list requests can bypass the LLM.
    Topical searches still go through the agent.
    """

    lower = " ".join(
        text.casefold().split()
    )

    if any(
        phrase in lower
        for phrase in (
            " then ",
            " and then ",
            " also ",
            " plus ",
        )
    ):
        return False

    if re.search(
        r"\b(create|add|delete|remove|"
        r"change|update|move|mark|complete|set)\b",
        lower,
    ):
        return False

    is_question = bool(
        re.match(
            r"\s*(show|list|display|what are|"
            r"which are|find all|how many)\b",
            lower,
        )
    )

    mentions_tasks = bool(
        re.search(
            r"\b(tasks?|to dos?|priorit(?:y|ies)|"
            r"highs|lows|mediums)\b",
            lower,
        )
    )

    topical = bool(
        re.search(
            r"\b(related to|about|similar to|"
            r"regarding|documentation|"
            r"authentication|presentation|api)\b",
            lower,
        )
    )

    return (
        is_question
        and mentions_tasks
        and not topical
    )