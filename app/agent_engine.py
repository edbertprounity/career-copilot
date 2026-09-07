"""Multi-turn tool-calling agent core for Student Career Copilot."""

from __future__ import annotations

import json
import re

from app.llm_client import (
    GeminiAPIError,
    GeminiJSONParseError,
    call_gemini_json,
    get_gemini_client,
)
from app.schemas import (
    AgentTraceStep,
    ExperienceRecord,
    ProjectRecord,
    StudentProfile,
)
from app.vector_store import search_top_jobs

MAX_TURNS = 15

KNOWN_TOOLS = (
    "update_profile_field",
    "inspect_completeness",
    "ask_targeted_follow_up",
    "ask_elaboration_question",
    "finalize_and_confirm_profile",
    "retrieve_matching_jobs",
)

UPDATABLE_FIELDS = (
    "name",
    "degree",
    "institution",
    "campus",
    "year",
    "technical_skills",
    "skill_evidence",
    "projects",
    "project_description",
    "experience",
    "experience_description",
    "activities",
    "target_roles",
    "links",
)

# Index-addressed writes: replace one record's description without resending the
# whole list, which is how sibling entries used to get lost.
ITEM_DESCRIPTION_FIELDS = {
    "project_description": "projects",
    "experience_description": "experience",
}

STRING_FIELDS = ("name", "degree", "institution", "campus", "year")

LIST_STR_FIELDS = ("technical_skills", "activities", "target_roles")

REQUIRED_FIELD_CHECKS = (
    ("name", lambda p: bool(p.name.strip())),
    ("degree", lambda p: bool(p.degree.strip())),
    ("institution", lambda p: bool(p.institution.strip())),
    ("year", lambda p: bool(p.year.strip())),
    ("technical_skills", lambda p: bool(p.technical_skills)),
    ("projects", lambda p: bool(p.projects)),
    ("experience", lambda p: bool(p.experience)),
    ("target_roles", lambda p: bool(p.target_roles)),
)

GROUPED_FIELDS = ("name", "degree", "institution", "year")
INDIVIDUAL_FIELDS = ("technical_skills", "projects", "experience", "target_roles")

FOLLOW_UP_QUESTIONS = {
    "name": "What is your full name?",
    "degree": "What degree or qualification did you complete?",
    "institution": "Which institution did you study at?",
    "year": "In what year did you graduate, or what are your study years?",
    "technical_skills": "What technical skills would you like listed on your profile?",
    "projects": "What projects would you like to include?",
    "experience": "What work or internship experience should we include?",
    "target_roles": "What target roles are you aiming for?",
}

GROUPED_FOLLOW_UP = (
    "What's your name, and tell me about your degree — institution and year?"
)
_GROUPED_QUESTION_PARTS = {
    "name": "your full name",
    "degree": "your degree",
    "institution": "the institution",
    "year": "your graduation or study year",
}

GENERIC_FOLLOW_UP = (
    "Could you share a bit more about your background so I can complete your profile?"
)
ACTIVITIES_PROMPT = (
    "Any certifications, awards, or notable activities you'd like to include? "
    "(Optional — just say 'none' if not.)"
)
CONFIRM_PROMPT = (
    "Your required fields are complete. Does this look right? "
    "Reply confirm and I will retrieve matching jobs so you can pick one "
    "and generate a CV."
)

# A description under this many words carries too little for a CV bullet.
THIN_DESCRIPTION_WORDS = 8
UNTITLED_ITEM = "unnamed"
ELABORATION_QUESTION_TAIL = (
    "what was your specific role, and what did you build or accomplish?"
)
# User must say one of these to skip remaining thin entries and confirm.
_EXPLICIT_CONTINUE_RE = re.compile(
    r"\b(confirm( it| my profile)?|that's all|that is all|that's everything|"
    r"that is everything|skip( remaining| the rest)?|no more|nothing else|"
    r"retrieve( matching)? jobs|looks right)\b",
    re.I,
)

def update_profile_field(profile: StudentProfile, updates: dict) -> dict:
    """Write one or more top-level profile fields. Does not set is_confirmed."""
    if not isinstance(updates, dict):
        return {"status": "error", "reason": "updates must be an object"}

    updated_fields: list[str] = []
    errors: dict[str, str] = {}
    for field_name, value in updates.items():
        error = _apply_one_field(profile, str(field_name), value)
        if error:
            errors[str(field_name)] = error
        else:
            updated_fields.append(str(field_name))

    if errors:
        return {
            "status": "error",
            "updated_fields": updated_fields,
            "errors": errors,
        }
    if not updated_fields:
        return {"status": "error", "reason": "updates is empty"}
    return {"status": "success", "updated_fields": updated_fields}


def _apply_one_field(
    profile: StudentProfile, field_name: str, value: object
) -> str | None:
    """Apply one field write. Returns an error reason, or None on success."""
    if field_name not in UPDATABLE_FIELDS:
        return f"unknown field_name: {field_name}"

    if field_name in STRING_FIELDS:
        setattr(profile, field_name, "" if value is None else str(value))
        return None

    if field_name in LIST_STR_FIELDS:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return f"{field_name} value must be a list of strings"
        setattr(profile, field_name, [str(item) for item in value])
        return None

    if field_name == "skill_evidence":
        parsed_evidence, error = _parse_skill_evidence(value)
        if error:
            return error
        profile.skill_evidence = parsed_evidence
        return None

    if field_name == "projects":
        parsed_projects, error = _parse_record_list(value, ProjectRecord, "projects")
        if error:
            return error
        profile.projects = parsed_projects
        return None

    if field_name == "experience":
        parsed_experience, error = _parse_record_list(
            value, ExperienceRecord, "experience"
        )
        if error:
            return error
        profile.experience = parsed_experience
        return None

    if field_name in ITEM_DESCRIPTION_FIELDS:
        return _apply_item_description(profile, field_name, value)

    if field_name == "links":
        if not isinstance(value, dict):
            return "links value must be an object"
        merged = dict(profile.links)
        for key, link in value.items():
            if not isinstance(key, str) or not isinstance(link, str):
                return "links keys and values must be strings"
            merged[key] = link
        profile.links = merged
        return None

    return f"unknown field_name: {field_name}"


def _apply_item_description(
    profile: StudentProfile, field_name: str, value: object
) -> str | None:
    """Replace one project/experience description, leaving sibling entries intact."""
    if not isinstance(value, dict):
        return f"{field_name} value must be an object with index and description"
    if "index" not in value or "description" not in value:
        return f"{field_name} value must have index and description"
    index = value["index"]
    description = value["description"]
    if isinstance(index, bool) or not isinstance(index, int):
        return f"{field_name} index must be an integer"
    if not isinstance(description, str):
        return f"{field_name} description must be a string"
    items = getattr(profile, ITEM_DESCRIPTION_FIELDS[field_name])
    if not 0 <= index < len(items):
        return f"{field_name} index {index} is out of range"
    items[index].description = description
    return None


def _parse_skill_evidence(value: object) -> tuple[list[dict], str | None]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return [], "skill_evidence value must be a list of objects"
    parsed: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return [], f"skill_evidence[{index}] must be an object"
        if "skill" not in item or "source" not in item:
            return [], f"skill_evidence[{index}] must have skill and source"
        if not isinstance(item["skill"], str) or not isinstance(item["source"], str):
            return [], f"skill_evidence[{index}] skill and source must be strings"
        parsed.append({"skill": item["skill"], "source": item["source"]})
    return parsed, None


def _parse_record_list(
    value: object, model_cls: type, field_name: str
) -> tuple[list, str | None]:
    if isinstance(value, dict) or isinstance(value, model_cls):
        value = [value]
    if not isinstance(value, list):
        return [], f"{field_name} value must be a list of objects"
    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, model_cls):
            parsed.append(item)
            continue
        if not isinstance(item, dict):
            return [], f"{field_name}[{index}] must be an object matching {model_cls.__name__}"
        try:
            parsed.append(model_cls(**item))
        except Exception as exc:
            return [], (
                f"{field_name}[{index}] is not a valid {model_cls.__name__}: {exc}"
            )
    return parsed, None


def inspect_completeness(profile: StudentProfile) -> dict:
    """Return which required profile fields are still empty. Pure Python."""
    missing_fields = [
        name for name, is_present in REQUIRED_FIELD_CHECKS if not is_present(profile)
    ]
    return {"missing_fields": missing_fields, "complete": not missing_fields}


def detect_thin_content(profile: StudentProfile) -> list[dict]:
    """Flag project/experience entries whose description is too thin for a CV.

    A description under THIN_DESCRIPTION_WORDS words is thin, an empty one
    included. Pure Python — no LLM judgement.
    """
    thin: list[dict] = []
    for index, project in enumerate(profile.projects):
        if _is_thin_description(project.description):
            thin.append(
                {
                    "type": "project",
                    "index": index,
                    "title_or_role": project.title.strip() or UNTITLED_ITEM,
                }
            )
    for index, item in enumerate(profile.experience):
        if _is_thin_description(item.description):
            thin.append(
                {
                    "type": "experience",
                    "index": index,
                    "title_or_role": item.role.strip() or UNTITLED_ITEM,
                }
            )
    return thin


def _is_thin_description(description: str) -> bool:
    return len(description.split()) < THIN_DESCRIPTION_WORDS


def ask_elaboration_question(item_type: str, item_title: str) -> dict:
    """Return one elaboration question for a thin entry. Deterministic template."""
    type_text = str(item_type).strip() or "entry"
    title_text = str(item_title).strip() or UNTITLED_ITEM
    return {
        "question": (
            f"Can you tell me more about your {title_text} {type_text} — "
            f"{ELABORATION_QUESTION_TAIL}"
        )
    }


def ask_targeted_follow_up(missing_fields: list[str]) -> dict:
    """Return one follow-up question. Groups short identity/education fields."""
    if not missing_fields:
        return {"question": GENERIC_FOLLOW_UP}
    if missing_fields[0] in GROUPED_FIELDS:
        grouped = [field for field in GROUPED_FIELDS if field in missing_fields]
        return {"question": _grouped_follow_up_question(grouped)}
    field = missing_fields[0]
    question = FOLLOW_UP_QUESTIONS.get(
        field, f"Could you provide your {field}?"
    )
    return {"question": question}


def _grouped_follow_up_question(grouped: list[str]) -> str:
    if not grouped:
        return GENERIC_FOLLOW_UP
    if grouped == list(GROUPED_FIELDS):
        return GROUPED_FOLLOW_UP
    if len(grouped) == 1:
        return FOLLOW_UP_QUESTIONS.get(
            grouped[0], f"Could you provide your {grouped[0]}?"
        )
    labels = [_GROUPED_QUESTION_PARTS[field] for field in grouped]
    if len(labels) == 2:
        return f"Could you share {labels[0]} and {labels[1]}?"
    return f"Could you share {', '.join(labels[:-1])}, and {labels[-1]}?"


def finalize_and_confirm_profile(profile: StudentProfile) -> dict:
    """Confirm only if target_roles is set and all required fields are filled."""
    if not profile.target_roles:
        return {"status": "error", "reason": "target_roles required"}
    completeness = inspect_completeness(profile)
    if not completeness["complete"]:
        return {
            "status": "error",
            "reason": "profile incomplete",
            "missing_fields": completeness["missing_fields"],
        }
    profile.is_confirmed = True
    return {"status": "success"}


def retrieve_matching_jobs(profile: StudentProfile) -> dict:
    """Retrieve top matching jobs. Hard-gated on profile.is_confirmed."""
    if not profile.is_confirmed:
        return {"status": "error", "reason": "profile must be confirmed first"}
    return {"status": "success", "matches": search_top_jobs(profile)}


def run_agent_turn(
    profile: StudentProfile,
    conversation_history: list[dict],
    user_message: str,
    turn_number: int,
) -> AgentTraceStep:
    """Run one agent turn: LLM chooses a tool; Python executes it."""
    if turn_number >= MAX_TURNS and not profile.is_confirmed:
        return _step_limit_reached(profile, turn_number)

    before = profile.model_dump()
    client = get_gemini_client()

    if client is None:
        return _fallback_turn(
            profile,
            before,
            turn_number,
            conversation_history,
            thought=(
                "FALLBACK: Gemini client is unavailable (missing key or init "
                "failure). Asking the next profile question."
            ),
            parameters={},
        )

    try:
        parsed = call_gemini_json(
            client,
            system_instruction=_build_system_instruction(
                profile, conversation_history
            ),
            user_content=_format_user_content(conversation_history, user_message),
        )
    except (GeminiAPIError, GeminiJSONParseError) as exc:
        return _fallback_turn(
            profile,
            before,
            turn_number,
            conversation_history,
            thought=(
                f"FALLBACK: LLM response failed to parse or the API call failed "
                f"({exc}). Asking the next profile question."
            ),
            parameters={},
        )

    thought = str(parsed.get("thought") or "")
    tool_name = str(parsed.get("tool_name") or "").strip()
    parameters = parsed.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    assistant_message = str(parsed.get("assistant_message") or "")

    if tool_name != "none" and tool_name not in KNOWN_TOOLS:
        return _fallback_turn(
            profile,
            before,
            turn_number,
            conversation_history,
            thought=(
                f"FALLBACK: invalid tool_name {tool_name!r}. "
                "Asking the next profile question."
            ),
            parameters=parameters,
            assistant_message=assistant_message,
        )

    if (
        _user_confirms_profile(user_message, conversation_history)
        and inspect_completeness(profile)["complete"]
        and tool_name
        not in ("update_profile_field", "retrieve_matching_jobs", "finalize_and_confirm_profile")
    ):
        tool_name = "finalize_and_confirm_profile"
        parameters = {}

    if (
        tool_name == "finalize_and_confirm_profile"
        and not _user_skips_remaining_detail(user_message)
        and not _user_confirms_profile(user_message, conversation_history)
    ):
        target = _next_elaboration_target(profile, conversation_history)
        if target:
            tool_name = "ask_elaboration_question"
            parameters = {
                "item_type": target["type"],
                "item_title": target["title_or_role"],
            }

    if tool_name == "none":
        observation: dict = {"status": "no_tool"}
    else:
        observation = _dispatch_tool(profile, tool_name, parameters)

    if (
        tool_name == "finalize_and_confirm_profile"
        and isinstance(observation, dict)
        and observation.get("status") == "success"
        and (
            not assistant_message.strip()
            or CONFIRM_PROMPT in assistant_message
            or "confirmed" not in assistant_message.lower()
        )
    ):
        assistant_message = (
            "Your profile is confirmed. Pick a matching job to generate your CV."
        )

    # observation reflects the dispatched tool's raw result (e.g.
    # ask_targeted_follow_up's default question); assistant_message is the LLM's
    # own phrasing shown to the user, which may differ when the LLM has fuller
    # conversation context than the tool's simple missing-fields logic. This is
    # intentional — the tool's canned question is a fallback-path safety net,
    # not the primary UX.
    # If the model wrote fields but asked nothing, Python attaches the next
    # question (elaboration → activities → confirm) so the student is never
    # left staring at a thank-you with no next step.
    tool_name, observation, assistant_message = _apply_follow_on_prompt(
        profile, conversation_history, tool_name, observation, assistant_message
    )
    return AgentTraceStep(
        turn_number=turn_number,
        thought=thought,
        tool_name=tool_name,
        parameters=parameters,
        observation=observation,
        state_update=_profile_diff(before, profile.model_dump()),
        assistant_message=assistant_message,
    )


def _dispatch_tool(
    profile: StudentProfile, tool_name: str, parameters: dict
) -> dict:
    if tool_name == "update_profile_field":
        return update_profile_field(profile, parameters.get("updates"))

    if tool_name == "inspect_completeness":
        return inspect_completeness(profile)

    if tool_name == "ask_targeted_follow_up":
        missing_fields = parameters.get("missing_fields")
        if not isinstance(missing_fields, list):
            missing_fields = inspect_completeness(profile)["missing_fields"]
        return ask_targeted_follow_up([str(item) for item in missing_fields])

    if tool_name == "ask_elaboration_question":
        return ask_elaboration_question(
            str(parameters.get("item_type") or ""),
            str(parameters.get("item_title") or ""),
        )

    if tool_name == "finalize_and_confirm_profile":
        return finalize_and_confirm_profile(profile)

    if tool_name == "retrieve_matching_jobs":
        return retrieve_matching_jobs(profile)

    return {"status": "error", "reason": f"unknown tool: {tool_name}"}


def _step_limit_reached(
    profile: StudentProfile, turn_number: int
) -> AgentTraceStep:
    completeness = inspect_completeness(profile)
    missing_fields = completeness["missing_fields"]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        assistant_message = (
            f"The profile could not be completed in {MAX_TURNS} turns. "
            f"Still missing: {missing_text}."
        )
    else:
        assistant_message = (
            f"The profile could not be completed in {MAX_TURNS} turns."
        )
    observation = {
        "status": "step_limit_reached",
        **completeness,
    }
    return AgentTraceStep(
        turn_number=turn_number,
        thought=(
            f"HARD STOP: turn_number {turn_number} reached MAX_TURNS "
            f"({MAX_TURNS}) and the profile is not confirmed."
        ),
        tool_name="step_limit_reached",
        parameters={},
        observation=observation,
        state_update={},
        assistant_message=assistant_message,
        requires_input=False,
    )


def _fallback_turn(
    profile: StudentProfile,
    before: dict,
    turn_number: int,
    conversation_history: list[dict] | None,
    thought: str,
    parameters: dict,
    assistant_message: str = "",
) -> AgentTraceStep:
    follow_tool, question, extra = _next_actionable_prompt(
        profile, conversation_history
    )
    tool_name = follow_tool or "none"
    observation = dict(extra)
    if tool_name == "ask_targeted_follow_up":
        parameters = {
            "missing_fields": inspect_completeness(profile)["missing_fields"]
        }
    elif tool_name == "ask_elaboration_question":
        parameters = {
            "item_type": extra.get("item_type") or "",
            "item_title": extra.get("item_title") or "",
        }
    message = assistant_message or question or GENERIC_FOLLOW_UP
    if question and not _message_covers_prompt(assistant_message, question, extra):
        message = _with_follow_on_question(assistant_message, question)
    return AgentTraceStep(
        turn_number=turn_number,
        thought=thought,
        tool_name=tool_name,
        parameters=parameters,
        observation=observation,
        state_update=_profile_diff(before, profile.model_dump()),
        assistant_message=message,
    )


def _profile_diff(before: dict, after: dict) -> dict:
    return {
        key: {"before": before[key], "after": after[key]}
        for key in before
        if before[key] != after[key]
    }


def _format_user_content(
    conversation_history: list[dict], user_message: str
) -> str:
    lines: list[str] = []
    for item in conversation_history or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append(f"user: {user_message}")
    return "\n".join(lines)


def _grouped_missing_fields(missing_fields: list[str]) -> list[str]:
    return [field for field in GROUPED_FIELDS if field in missing_fields]


def _activities_prompt_already_asked(conversation_history: list[dict] | None) -> bool:
    for item in conversation_history or []:
        if item.get("role") != "assistant":
            continue
        if ACTIVITIES_PROMPT in str(item.get("content") or ""):
            return True
    return False


def _elaboration_already_asked(
    conversation_history: list[dict] | None, title_or_role: str
) -> bool:
    """True when an assistant turn already asked to elaborate on this entry."""
    needle = " ".join(title_or_role.lower().split())
    tail = ELABORATION_QUESTION_TAIL.lower()
    for item in conversation_history or []:
        if item.get("role") != "assistant":
            continue
        content = " ".join(str(item.get("content") or "").lower().split())
        if tail in content and needle and needle in content:
            return True
    return False


def _next_elaboration_target(
    profile: StudentProfile, conversation_history: list[dict] | None
) -> dict | None:
    """First thin entry not yet asked about, once every required field is filled."""
    if not inspect_completeness(profile)["complete"]:
        return None
    for item in detect_thin_content(profile):
        if not _elaboration_already_asked(conversation_history, item["title_or_role"]):
            return item
    return None


def _optional_activities_prompt_status(
    profile: StudentProfile, conversation_history: list[dict] | None
) -> str:
    if not inspect_completeness(profile)["complete"]:
        return "not_needed"
    if profile.activities:
        return "not_needed"
    if _activities_prompt_already_asked(conversation_history):
        return "already_asked"
    return "pending"


def _user_skips_remaining_detail(user_message: str) -> bool:
    return bool(_EXPLICIT_CONTINUE_RE.search(user_message or ""))


_AFFIRM_RE = re.compile(
    r"^\s*(yes|yeah|yep|y|ok|okay|sure|it is|yes it is|yes it looks right)\s*[.!]*\s*$",
    re.I,
)


def _last_assistant_content(conversation_history: list[dict] | None) -> str:
    for item in reversed(conversation_history or []):
        if item.get("role") == "assistant":
            return str(item.get("content") or "")
    return ""


def _user_confirms_profile(
    user_message: str, conversation_history: list[dict] | None
) -> bool:
    """True when the student accepted the confirm-to-retrieve prompt."""
    if _user_skips_remaining_detail(user_message):
        return True
    if not _AFFIRM_RE.match(user_message or ""):
        return False
    last = _last_assistant_content(conversation_history).lower()
    return "reply confirm" in last or CONFIRM_PROMPT.lower() in last


def _message_covers_prompt(message: str, question: str, extra: dict) -> bool:
    """True only when this turn already asked the computed next question."""
    msg = " ".join((message or "").lower().split())
    q = " ".join((question or "").lower().split())
    if q and q in msg:
        return True
    title = " ".join(str(extra.get("item_title") or "").lower().split())
    tail = ELABORATION_QUESTION_TAIL.lower()
    return bool(title and tail in msg and title in msg)


def _with_follow_on_question(existing: str, question: str) -> str:
    existing = (existing or "").strip()
    question = (question or "").strip()
    if not question:
        return existing
    if question in existing:
        return existing
    if not existing:
        return question
    return f"{existing}\n\n{question}"


def _next_actionable_prompt(
    profile: StudentProfile,
    conversation_history: list[dict] | None,
) -> tuple[str, str, dict]:
    """Python next question so a turn never ends with nowhere to go.

    Priority matches the system instruction: empty required field, then a thin
    project/experience description, then the optional activities prompt, then
    an explicit confirm-to-retrieve request. Confirmation and retrieve stay
    Python gates — this only asks the student what to do next.
    """
    if profile.is_confirmed:
        return "", "", {}

    missing = inspect_completeness(profile)["missing_fields"]
    if missing:
        observation = ask_targeted_follow_up(missing)
        return (
            "ask_targeted_follow_up",
            str(observation.get("question") or GENERIC_FOLLOW_UP),
            observation,
        )

    target = _next_elaboration_target(profile, conversation_history)
    if target:
        observation = ask_elaboration_question(
            target["type"], target["title_or_role"]
        )
        return (
            "ask_elaboration_question",
            str(observation["question"]),
            {
                **observation,
                "item_type": target["type"],
                "item_title": target["title_or_role"],
            },
        )

    if _optional_activities_prompt_status(profile, conversation_history) == "pending":
        return "none", ACTIVITIES_PROMPT, {"status": "optional_activities_prompt"}

    return "", CONFIRM_PROMPT, {"status": "awaiting_confirmation"}


def _apply_follow_on_prompt(
    profile: StudentProfile,
    conversation_history: list[dict] | None,
    tool_name: str,
    observation: dict,
    assistant_message: str,
) -> tuple[str, dict, str]:
    if profile.is_confirmed:
        return tool_name, observation, assistant_message
    if tool_name in {
        "finalize_and_confirm_profile",
        "retrieve_matching_jobs",
        "step_limit_reached",
    }:
        return tool_name, observation, assistant_message

    follow_tool, question, extra = _next_actionable_prompt(
        profile, conversation_history
    )
    if not question:
        return tool_name, observation, assistant_message
    if _message_covers_prompt(assistant_message, question, extra):
        return tool_name, observation, assistant_message

    merged = dict(observation) if isinstance(observation, dict) else {}
    merged["follow_on"] = {"tool_name": follow_tool or "awaiting_confirmation", **extra}
    return tool_name, merged, _with_follow_on_question(assistant_message, question)


def _follow_up_state_block(
    profile: StudentProfile, conversation_history: list[dict] | None
) -> str:
    missing = inspect_completeness(profile)["missing_fields"]
    grouped = _grouped_missing_fields(missing)
    next_individual = next(
        (field for field in missing if field in INDIVIDUAL_FIELDS), None
    )
    return (
        f"grouped_missing: {json.dumps(grouped)}\n"
        f"next_individual_field: {json.dumps(next_individual)}\n"
        f"required_complete: {json.dumps(not missing)}\n"
        f"thin_content: {json.dumps(detect_thin_content(profile))}\n"
        f"next_elaboration: {json.dumps(_next_elaboration_target(profile, conversation_history))}\n"
        f"optional_activities_prompt: {_optional_activities_prompt_status(profile, conversation_history)}\n"
    )


def _build_system_instruction(
    profile: StudentProfile,
    conversation_history: list[dict] | None = None,
) -> str:
    profile_json = json.dumps(profile.model_dump(), ensure_ascii=True)
    follow_up_state = _follow_up_state_block(profile, conversation_history)
    return f"""You are the Student Career Copilot agent. You collect a confirmed student profile by choosing tools. You never decide completeness or confirmation yourself — Python functions do that.

Current StudentProfile JSON:
{profile_json}

Follow-up state (computed by Python — trust this over guessing from the JSON):
{follow_up_state}

You must reply with this exact JSON object shape and no other keys:
{{"thought": str, "tool_name": str, "parameters": dict, "assistant_message": str}}

tool_name must be one of these exact names, or the literal "none":
1. update_profile_field
   parameters: {{"updates": {{field_name: value, ...}}}}
   Each field_name must be one of: name, degree, institution, campus, year, technical_skills, skill_evidence, projects, project_description, experience, experience_description, activities, target_roles, links
   For name, degree, institution, campus, year: value is a string.
   For technical_skills, activities, target_roles: value is a list of strings (target_roles is a list, not a single string).
   For skill_evidence: value is a list of objects {{"skill": str, "source": str}}.
   For projects: value is a list of objects {{"title": str, "context": str, "description": str, "technologies": [str, ...]}}.
   For experience: value is a list of objects {{"role": str, "organisation": str, "type": str, "description": str}}.
   For project_description: value is an object {{"index": int, "description": str}}. It replaces the description of projects[index] and leaves every other project untouched.
   For experience_description: value is an object {{"index": int, "description": str}}. Same, for experience[index].
   When the user adds detail about a project or experience entry that is ALREADY in the profile, use project_description / experience_description with that entry's index (from thin_content, next_elaboration, or the profile JSON). Replace the old description with the fuller text — do not append to it, and do not resend a whole projects or experience list, because sibling entries can be lost that way. Use the whole-list projects / experience fields only when adding a new entry.
   For links: value is an object of string keys to string URLs; new keys are merged into existing links.
   If the user's message contains multiple distinct facts the user has actually committed to, include ALL of those committed facts in a single update_profile_field call as a dict of field_name -> value pairs — do not pick only one and discard the rest. Do not write a field the user is still deciding.
2. inspect_completeness
   parameters: {{}}
   Use ONLY when you need to silently check internal state before deciding what to do next. It does not communicate anything to the user by itself. Do not call it when you are about to ask the user a question.
3. ask_targeted_follow_up
   parameters: {{"missing_fields": [str, ...]}}
   Required empty-field names: name, degree, institution, year, technical_skills, projects, experience, target_roles.
   GROUPED_FIELDS = name, degree, institution, year. INDIVIDUAL_FIELDS = technical_skills, projects, experience, target_roles.
   If grouped_missing is non-empty, pass ALL of those grouped names in missing_fields (not just the first) and set assistant_message to ONE combined question covering them, e.g. "{GROUPED_FOLLOW_UP}"
   If grouped_missing is empty and next_individual_field is set, pass that individual field first (other remaining missing names may follow) and ask only that one richer question. Do not batch individual fields into one question.
   Use this whenever required fields are still empty AND you are about to ask the user a question about what's missing. This is your DEFAULT choice whenever required fields are still empty and the user's last message did not just supply new facts (for example "what else do you need to know?", "what next?", "go on", or other vague replies). Do not just write a required-field question in assistant_message with tool_name "none".
   Do not use this tool for certifications/awards/activities — that field is optional and is handled separately below.
4. ask_elaboration_question
   parameters: {{"item_type": "project" or "experience", "item_title": str}}
   The entry exists but its description is too thin to be useful in a CV. Use this ONLY when next_elaboration is not null: pass its "type" as item_type and its "title_or_role" as item_title, and set assistant_message to exactly "Can you tell me more about your <title_or_role> <type> — {ELABORATION_QUESTION_TAIL}"
   Priority, in order: (a) a required field is completely empty, so use ask_targeted_follow_up; (b) required_complete is true and next_elaboration is not null, so use this tool; (c) nothing is missing or thin, so continue with the optional activities prompt or confirmation described below.
   Never use this for an empty required field, and never ask about the same entry twice — next_elaboration is already null once that entry has been asked about.
5. finalize_and_confirm_profile
   parameters: {{}}
   Call this only when target_roles is set AND all other required fields are filled. Python will reject confirmation otherwise and tell you what's still missing — if rejected, call ask_targeted_follow_up next using the returned missing_fields.
   Do not call this on the same turn as the one-time optional activities prompt. After that prompt has been asked (optional_activities_prompt is already_asked), or if the user says "none"/"skip"/similar, or explicitly asks you to confirm, call this tool. The optional prompt must never block confirmation.
6. retrieve_matching_jobs
   parameters: {{}}
   Only useful after the profile is confirmed. Python will reject this if is_confirmed is false.
   On success it returns {{"status": "success", "matches": [{{"document_id": str, "title": str, "similarity_score": float, "supporting_excerpt": str, "gap_analysis": str}}, ...]}} — the top 3 jobs by local embedding cosine similarity. supporting_excerpt is a sentence taken from that job's real description. gap_analysis is 1-2 sentences comparing the confirmed profile to that job description.
7. none
   parameters: {{}}
   Use when you only need to reply and should not change state. This is also the tool for the one-time optional activities prompt (not ask_targeted_follow_up).

Example — grouped identity/education fields are missing (do this, not four separate questions):
user: "hi, I'd like to build my profile"
correct: {{"thought": "grouped_missing lists name, degree, institution, and year, so I ask them together.", "tool_name": "ask_targeted_follow_up", "parameters": {{"missing_fields": ["name", "degree", "institution", "year"]}}, "assistant_message": "{GROUPED_FOLLOW_UP}"}}
wrong: asking only for name, or tool_name "none" with a question, or inspect_completeness.

Example — user asks what is needed and only individual fields remain:
user: "What else do you need to know?"
correct: {{"thought": "Grouped fields are filled; next_individual_field is technical_skills.", "tool_name": "ask_targeted_follow_up", "parameters": {{"missing_fields": ["technical_skills", "projects", "experience", "target_roles"]}}, "assistant_message": "What technical skills would you like listed on your profile?"}}
wrong: tool_name "inspect_completeness", or a prose-only question with tool_name "none".

Example — user is undecided about a field (do this, not update_profile_field with guessed options):
user: "I built a dashboard at UTS but I am still deciding between data and software roles."
correct: {{"thought": "Degree/institution facts are stated, but target_roles is uncertain, so I must not write a guessed list. Ask them to clarify the role.", "tool_name": "ask_targeted_follow_up", "parameters": {{"missing_fields": ["target_roles"]}}, "assistant_message": "Would you like to target both Data and Software roles, or pick one for now?"}}
wrong: tool_name "update_profile_field" with target_roles ["Data Roles", "Software Roles"] (or any guessed list) just because they mentioned both options.

Example — required fields are complete but one project description is too thin, so next_elaboration is {{"type": "project", "index": 0, "title_or_role": "Course planner"}} (this is NOT ask_targeted_follow_up):
user: "that's everything I have"
correct: {{"thought": "required_complete is true and next_elaboration points at projects[0], so I ask for detail on that entry before moving on.", "tool_name": "ask_elaboration_question", "parameters": {{"item_type": "project", "item_title": "Course planner"}}, "assistant_message": "Can you tell me more about your Course planner project — {ELABORATION_QUESTION_TAIL}"}}
wrong: ask_targeted_follow_up (no required field is empty). wrong: the optional activities prompt or finalize_and_confirm_profile while next_elaboration is not null.

Example — the user answers that elaboration question (replace that one description; never resend the list):
user: "I built the timetable clash checker in Flask, wrote the SQL queries and the unit tests for it."
correct: {{"thought": "This is fuller detail for projects[0], so I write it with project_description and the other projects stay untouched.", "tool_name": "update_profile_field", "parameters": {{"updates": {{"project_description": {{"index": 0, "description": "Built the timetable clash checker in Flask, wrote the SQL queries and the unit tests for it."}}}}}}, "assistant_message": "Thanks — I've added that detail to your Course planner project."}}
wrong: a whole "projects" list (sibling entries can be lost). wrong: appending the new text to the old thin description.

Example — required fields are complete, activities is empty, optional_activities_prompt is pending (do this BEFORE confirm; this is NOT ask_targeted_follow_up):
user: "that's everything I have" or "go on"
correct: {{"thought": "Required fields are complete. Offer the one-time optional activities prompt before confirm.", "tool_name": "none", "parameters": {{}}, "assistant_message": "{ACTIVITIES_PROMPT}"}}
wrong: ask_targeted_follow_up (activities is not a required missing field). wrong: finalize_and_confirm_profile on this same turn.

Example — after that optional prompt, the user declines:
user: "none"
correct: {{"thought": "User skipped optional activities. Required fields are complete, so I confirm.", "tool_name": "finalize_and_confirm_profile", "parameters": {{}}, "assistant_message": "Your profile is confirmed."}}

Rules:
- thought is your brief reasoning.
- assistant_message is what the student should see.
- Choose exactly one tool and its parameters. Do not compute completeness or set is_confirmed yourself.
- Never invent profile facts the user has not stated.
- If the user's message expresses uncertainty or presents multiple options without committing to one (words like "deciding between", "not sure", "either ... or", "maybe", "have not yet decided"), do NOT write a list of guessed values into update_profile_field for that field. Treat it as incomplete: call ask_targeted_follow_up and ask them to clarify which single option they actually want, or to confirm both are genuinely intended. Because you may choose only one tool, prefer this clarifying follow-up over writing the other committed facts in the same turn.
- When optional_activities_prompt is pending, ask it once using tool_name "none" and the exact assistant_message shown in the example. Do not ask it again when the status is already_asked or not_needed. If next_elaboration is not null on that turn, ask the elaboration question first and leave the optional prompt for a later turn.
- A thin description must never deadlock the profile: if the user declines to add detail, gives a vague reply again, or explicitly asks you to confirm, continue with the optional activities prompt or finalize_and_confirm_profile as normal.
"""
