"""Local tool permission policy."""

import json

from pathlib import Path
from .config import PERMISSIONS_FILE


VALID_MODES = {
    "enabled",
    "approval_required",
    "disabled"
}


def load_permissions(
    path: Path = PERMISSIONS_FILE
) -> dict[str, str]:

    with Path(path).open(
        encoding="utf-8"
    ) as file:
        policy = json.load(file)

    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "Tool permissions must be a non-empty JSON object"
        )

    if any(
        not isinstance(key, str)
        or mode not in VALID_MODES
        for key, mode in policy.items()
    ):
        raise ValueError(
            "Invalid tool name or permission mode"
        )

    return policy


def get_permission(
    tool_name: str,
    policy: dict[str, str] | None = None
) -> str:

    permissions = (
        policy if policy is not None
        else load_permissions()
    )

    return permissions.get(
        tool_name,
        "disabled"
    )


def require_enabled(
    tool_name: str,
    policy: dict[str, str] | None = None
) -> None:

    mode = get_permission(
        tool_name,
        policy
    )

    if mode != "enabled":
        raise PermissionError(
            f"{tool_name!r} is {mode}; execution blocked"
        )