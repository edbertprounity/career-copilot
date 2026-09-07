"""Pydantic models for the Student Career Copilot agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectRecord(BaseModel):
    title: str = ""
    context: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ExperienceRecord(BaseModel):
    role: str = ""
    organisation: str = ""
    type: str = ""
    description: str = ""


class StudentProfile(BaseModel):
    document_id: str = ""
    name: str = ""
    degree: str = ""
    institution: str = ""
    campus: str = ""
    year: str = ""
    technical_skills: list[str] = Field(default_factory=list)
    skill_evidence: list[dict] = Field(default_factory=list)
    projects: list[ProjectRecord] = Field(default_factory=list)
    experience: list[ExperienceRecord] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(default_factory=dict)
    is_confirmed: bool = False


class AgentTraceStep(BaseModel):
    turn_number: int
    thought: str
    tool_name: str
    parameters: dict
    observation: dict
    state_update: dict
    assistant_message: str
    requires_input: bool = True
