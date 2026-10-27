import re
from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

Status = Literal["To do", "In progress", "Done"]
Priority = Literal["Low", "Medium", "High"]


def clean_title(value: str) -> str:
    value = value.strip()

    if not value:
        raise ValueError("Task title cannot be empty")

    return value


def validate_due_date(value):
    if value is None or isinstance(value, date):
        return value

    if (
        not isinstance(value, str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
    ):
        raise ValueError("due_date must use YYYY-MM-DD")

    return date.fromisoformat(value)


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    status: Status = "To do"
    priority: Priority = "Medium"
    due_date: date | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value):
        return clean_title(value)

    @field_validator("due_date", mode="before")
    @classmethod
    def check_date(cls, value):
        return validate_due_date(value)


class TaskUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    status: Status | None = None
    priority: Priority | None = None
    due_date: date | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value):
        return clean_title(value) if value is not None else value

    @field_validator("due_date", mode="before")
    @classmethod
    def check_date(cls, value):
        return validate_due_date(value)

    @field_validator(
        "title", "description", "status", "priority"
    )
    @classmethod
    def dont_clear_required_fields(cls, value):
        if value is None:
            raise ValueError("This field cannot be null")
        return value