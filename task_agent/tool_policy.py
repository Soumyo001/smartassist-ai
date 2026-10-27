import json
from pathlib import Path

from .storage import BASE_DIR


PERMISSIONS_PATH = (
    BASE_DIR / "config" / "tool_permissions.json"
)

KNOWN_TOOLS = frozenset({
    "create_task",
    "list_tasks",
    "get_task",
    "update_task",
    "complete_task",
    "delete_task",
    "search_tasks",
})


def get_enabled_tool_names(
    path: Path = PERMISSIONS_PATH
) -> frozenset[str]:

    config = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    names = config.get("enabled_tools")

    if (
        not isinstance(names, list)
        or any(not isinstance(n, str) for n in names)
    ):
        raise ValueError(
            "enabled_tools must be an array of tool names"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate tool names in allowlist"
        )

    unknown = set(names) - KNOWN_TOOLS

    if unknown:
        raise ValueError(
            f"Unknown tool(s): {sorted(unknown)}"
        )

    return frozenset(names)


def require_allowed(
    name: str,
    path: Path = PERMISSIONS_PATH
) -> None:

    if name not in get_enabled_tool_names(path):
        raise PermissionError(
            f"Tool {name!r} is not allowed"
        )