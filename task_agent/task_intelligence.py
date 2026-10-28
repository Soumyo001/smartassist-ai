"""Deterministic task identification and clarification helpers."""

import re


def normalized_title(value: str) -> str:
    return " ".join(
        re.findall(r"\w+", value.casefold())
    )


def mentioned_task_ids(text: str) -> set[int]:
    """
    Extract explicitly referenced task IDs.

    Supports:
      #17, $17, task 17, task #17
      ID 17, task ID 17
      task ID 1,2,3
      task #17 and #18

    Does not interpret ISO dates as task IDs.
    """

    anchor = (
        r"(?<!\w)(?:[#\$]\s*|"
        r"\b(?:task\s+)?ids?\s*[#\$]?\s*|"
        r"\btask\s*[#\$]?\s*)"
        r"(\d+)(?![\w-])"
    )

    continuation = (
        r"\s*(?:,|\band\b)\s*"
        r"(?:(?:task\s+)?ids?\s*|task\s*)?"
        r"[#\$]?\s*(\d+)(?![\w-])"
    )

    found = set()

    for match in re.finditer(anchor, text, re.I):
        found.add(int(match.group(1)))

        position = match.end()

        # Additional IDs must follow an explicit
        # reference and a list separator.
        while True:
            next_match = re.match(
                continuation,
                text[position:],
                re.I,
            )

            if next_match is None:
                break

            found.add(int(next_match.group(1)))
            position += next_match.end()

    return found


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
            r"\s*(show|list|display|sort|order|"
            r"what are|which are|find all|how many)\b",
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

def is_sort_correction(message: str) -> bool:
    """Detect feedback about an incorrectly ordered list."""

    words = " ".join(
        re.findall(r"\w+", message.casefold())
    )

    # A question about the previous order is not a
    # request to replay the previous list.
    if message.strip().endswith("?") and re.match(
        r"^\s*(was|were|is|are|did|does|do|why|how)\b",
        message,
        re.I,
    ):
        return False

    # A new sorting instruction is not correction feedback.
    if re.match(
        r"^(show|list|display|sort|order)\b",
        words,
    ):
        return False

    return bool(
        re.search(
            r"\b(wrong|incorrect|not|failed|"
            r"isn|wasn|didn)\b",
            words,
        )
        and re.search(
            r"\b(order|ordered|ordere|"
            r"sort|sorted|sorting|sequence)\b",
            words,
        )
    )


def is_merge_request(text: str) -> bool:
    """A merge must never be silently converted into a delete."""
    return bool(re.search(
        r"\b(merge|merging|merged|combine|combining|consolidate)\b",
        text,
        re.I,
    ))


def is_explicit_delete_request(text: str) -> bool:
    """Only explicit delete/remove language authorizes deletion."""
    return bool(re.search(
        r"\b(delete|remove)\b",
        text,
        re.I,
    ))


def is_sort_audit_question(text: str) -> bool:
    """Detect questions about the accuracy of a previous sort."""
    words = " ".join(
        re.findall(r"\w+", text.casefold())
    )

    # Correction statements belong to the correction
    # handler, which rebuilds the previous task list.
    if is_sort_correction(text):
        return False

    return bool(
        re.search(
            r"\b(was|were|is|did|which|what)\b",
            words,
        )
        and re.search(
            r"\b(order|ordered|sorted|sorting|sequence)\b",
            words,
        )
        and re.search(
            r"\b(last|previous|prior|earlier|that|"
            r"you|response|list|tasks)\b",
            words,
        )
    )



def audit_last_priority_list(messages: list[str]) -> str:
    """Verify the most recent actual priority-sorted task list."""

    for message in reversed(messages):

        # Match a real list heading, not a conversational
        # sentence containing "sorted by priority".
        header = re.search(
            r"(?m)^[ \t]*\*?Sorted by priority:"
            r"[ \t]*([^\r\n*]+)\*?[ \t]*$",
            message,
            re.I,
        )

        task_heading = re.search(
            r"\*\*Your tasks \((\d+)\)\*\*",
            message,
        )

        if not header or not task_heading:
            continue

        priorities = [
            name.title()
            for name in re.findall(
                r"\b(High|Medium|Low)\b",
                header.group(1),
                re.I,
            )
        ]

        if (
            len(priorities) != 3
            or len(set(priorities)) != 3
        ):
            return (
                "The previous priority order "
                "could not be verified."
            )

        actual = [
            name.title()
            for name in re.findall(
                r"(?m)^[ \t]*"
                r"(?:To do|In progress|Done)"
                r"[ \t]*·[ \t]*"
                r"(High|Medium|Low)"
                r"[ \t]+priority\b",
                message,
                re.I,
            )
        ]

        label = " → ".join(priorities)
        expected = int(task_heading.group(1))

        if expected == 0:
            return (
                "The previous task list was empty, "
                "so no ordering was needed."
            )

        if len(actual) != expected:
            return (
                f"The last list declared **{label}**, "
                "but I couldn't verify all its task rows."
            )

        positions = [
            priorities.index(name)
            for name in actual
        ]

        correctly_sorted = all(
            left <= right
            for left, right in zip(
                positions,
                positions[1:],
            )
        )

        if correctly_sorted:
            return (
                "**Yes.** I verified the task rows "
                f"follow **{label}**."
            )

        return (
            "**No.** The task rows do not follow "
            f"the declared **{label}** order."
        )

    return (
        "I couldn't find a previous priority-sorted "
        "task list to verify."
    )


def is_retryable_groq_json_error(error: Exception) -> bool:
    """Recognize specific malformed structured-output errors."""

    if type(error).__name__ != "BadRequestError":
        return False

    body = getattr(error, "body", None)
    code = getattr(error, "code", None)

    if isinstance(body, dict):
        details = body.get("error", body)

        if isinstance(details, dict):
            code = details.get("code", code)

    return (
        code in {
            "json_validate_failed",
            "tool_use_failed",
        }
        or any(
            value in str(error)
            for value in (
                "json_validate_failed",
                "tool_use_failed",
            )
        )
    )

def standalone_due_date(
    text: str,
    today,
) -> str | None:
    """
    Parse a standalone English date reply.

    Only accepts an unambiguous day and month.
    Does not silently guess next year for a past date.
    """

    import calendar
    from datetime import date

    month_pattern = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )

    match = re.fullmatch(
        rf"\s*(\d{{1,2}})(?:st|nd|rd|th)?"
        rf"\s+({month_pattern})"
        r"(?:\s+(\d{4}))?\s*[.!]?\s*",
        text,
        re.I,
    )

    if not match:
        return None

    token = match.group(2).casefold()

    month = next(
        (
            number
            for number in range(1, 13)
            if calendar.month_name[number]
            .casefold()
            .startswith(token[:3])
        ),
        None,
    )

    if month is None:
        return None

    explicit_year = match.group(3) is not None

    year = (
        int(match.group(3))
        if explicit_year
        else today.year
    )

    try:
        parsed = date(
            year,
            month,
            int(match.group(1)),
        )
    except ValueError:
        return None

    if not explicit_year and parsed < today:
        return None

    return parsed.isoformat()
