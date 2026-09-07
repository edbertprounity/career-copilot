# Prompt record

Sequential log of every build prompt for Student Career Copilot.

## Entry 0 — Initial project scaffolding
**Goal**: Create the empty-workspace skeleton (folders, placeholders, requirements, config template, gitignore, README, and this prompt log) without a real `.env` or any backend modules.
**Plan proposed**: Create `app/` and `data/` with `.gitkeep`, `docs/prompt_record.md`, `README.md`, `streamlit_app.py` placeholder, `requirements.txt` (unpinned listed packages), `config.example` with `GEMINI_API_KEY=your-key-here`, and `.gitignore` excluding `.env` and venv. No `git init`, no venv, no package install, no Gemini calls. Verify by listing the tree, confirming no `.env`, confirming `app/` has no `.py` modules, and running `python -m py_compile streamlit_app.py`.
**Files changed**: `app/.gitkeep`, `data/.gitkeep`, `docs/prompt_record.md`, `README.md`, `streamlit_app.py`, `requirements.txt`, `config.example`, `.gitignore`
**Verification performed**: Confirmed workspace root `D:\Study\UTS\Advanced AI\career-copilot`. Listed created files via workspace search and file reads. Confirmed no `.env` exists. Confirmed `config.example` is exactly `GEMINI_API_KEY=your-key-here`. Confirmed `app/` has no `.py` modules. Confirmed this file has Entry 0. Ran `python -m py_compile streamlit_app.py` from the project root.
**Result**: Pass. `py_compile` exited 0 with no output. All planned files present. No `.env`. No backend modules.
**Notes**: Assumptions approved with the go-ahead: `.gitkeep` in empty folders; no git init; no version pins; Streamlit file is a no-op stub; uvicorn/streamlit run commands documented as future; Gemini SDK check deferred until first live call. No `.env` created. A full recursive `Get-ChildItem` listing was not run — auto-review blocked a wide inventory; verification used targeted file reads plus the compile check instead.

## Entry 1 — Shared Gemini client module
**Goal**: Add `app/llm_client.py` only (cached client, JSON generate_content helper, `__main__` smoke test), plus standing README/prompt-log updates. Create a project-local venv first and install `google-genai` and `python-dotenv` into it.
**Plan proposed**: Create `app/llm_client.py` with `get_gemini_client()` (dotenv + cached `genai.Client`, never raises) and `call_gemini_json()` (`response_mime_type="application/json"`, default model `gemini-2.5-flash`, specific `GeminiAPIError` / `GeminiJSONParseError`). Do not switch models on failure. Update README structure/phase. Create `venv/` at this project root, activate it, install the two packages, then compile and run `python -m app.llm_client`.
**Files changed**: `app/llm_client.py` (created), `README.md` (structure + Phase 1), `docs/prompt_record.md` (this entry). Created project-local `venv/` (gitignored). Did not modify other application files. Did not read or write `.env`.
**Verification performed**: `python -m venv venv` at project root. Confirmed venv python is `D:\Study\UTS\Advanced AI\career-copilot\venv\Scripts\python.exe`. Activated that venv and ran `python -m pip install --upgrade google-genai python-dotenv`. `pip show google-genai` → 2.20.0. Key presence check printed `GEMINI_API_KEY=SET` (value not printed). `python -m py_compile app/llm_client.py`. Live: `python -m app.llm_client`.
**Result**: Partial. Compile passed (exit 0). Live call failed (exit 1) with a readable error, not a traceback: `404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.', 'status': 'NOT_FOUND'}}`. Default model left as `gemini-2.5-flash` (no silent switch).
**Notes**: SDK 2.20.0 is current. Key was present and not rejected for prefix. The SDK also printed a warning about automatic function calling in `Models.generate_content`; flagged only, not changed. Proposed follow-up (awaiting approval): change the default `model` to `gemini-3.6-flash` as named in the API error.

## Entry 2 — Default Gemini model to 3.6-flash
**Goal**: Change the default `call_gemini_json` model from `gemini-2.5-flash` to `gemini-3.6-flash` after the 404, then re-run the smoke test.
**Plan proposed**: One-line default-model change in `app/llm_client.py`; re-run `python -m app.llm_client`. No other application files. No silent fallback if this model also fails.
**Files changed**: `app/llm_client.py` (default `model` only), `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m app.llm_client` from project root using `D:\Study\UTS\Advanced AI\career-copilot\venv\Scripts\python.exe`
**Result**: Pass. Exit 0. Output included the existing AFC warning, then `SUCCESS: {'status': 'ok'}`.
**Notes**: File structure unchanged, so README was not edited. AFC warning still present; still flagged only, not changed.

## Entry 3 — Agent core schemas and tool dispatch
**Goal**: Add `app/schemas.py` and `app/agent_engine.py` (five tools + `run_agent_turn`) without step/turn limits or real job retrieval. Put `assistant_message` on `AgentTraceStep` itself; keep `state_update` as profile before/after diffs only.
**Plan proposed**: Nested `Contact`/`Education` plus `StudentProfile`/`AgentTraceStep`. Deterministic tools for update, completeness, follow-up, finalize (`target_job` hard gate), and retrieve (confirm gate + stub). `run_agent_turn` uses `call_gemini_json` and falls back to `ask_targeted_follow_up` on invalid tool/parse failure. Docs: README Phase 2 + this entry. No `main.py` / Streamlit / vector store / CV / turn limit.
**Files changed**: `app/schemas.py` (created), `app/agent_engine.py` (created), `README.md` (structure + Phase 2), `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/schemas.py app/agent_engine.py`. Deterministic in-process checks for inspect/finalize/retrieve/update/follow-up. `pip show google-genai` → 2.20.0. Key presence `SET` (value not printed). Live `run_agent_turn` with message "Hi, my name is Alex Tan and I am targeting a data analyst role."
**Result**: Pass. Deterministic checks printed `DETERMINISTIC_CHECKS=PASS`. Live turn: `tool_name=update_profile_field`, observation success on `contact`, `state_update` only `{contact: before/after}`, `assistant_message` set, `TRACE_OK=True`, `is_confirmed=False`. Compile and live call exited 0. SDK AFC warning still printed.
**Notes**: One tool per turn, so this live call wrote `contact.name` only and left `target_job` unset — expected, not a bug. README still says `target_roles` in the constraints section while the code uses `target_job`; flagged only, not rewritten. AFC warning still flagged only. `year` is `str`. `conversation_history` is `list[dict]` with `role`/`content`.

## Entry 4 — Confirm tool-calling mechanism
**Goal**: Confirm whether `run_agent_turn` uses Gemini native function-calling / AFC, or JSON `tool_name` parsing plus Python dispatch.
**Plan proposed**: Read-only confirmation; no application code changes. Append this log entry only.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: Read `app/llm_client.py` `call_gemini_json` and `app/agent_engine.py` `run_agent_turn` / `_dispatch_tool`.
**Result**: Confirmed: tool selection is JSON `tool_name` string parsing + Python if/dispatch. Native AFC is not used. `generate_content` config is only `system_instruction` and `response_mime_type="application/json"` — no `tools`, no function declarations, no AFC config.
**Notes**: The SDK AFC warning can still print even though this project does not pass tool declarations. No code change.

## Entry 5 — README target_job wording
**Goal**: Change the README constraints line from `target_roles` to `target_job` so it matches `StudentProfile` and `finalize_and_confirm_profile`.
**Plan proposed**: One-word README correction; append this log entry. No application code changes.
**Files changed**: `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: Search `README.md` for `target_roles` after the edit.
**Result**: Pass. The constraints line now reads `target_job`. No remaining `target_roles` in `README.md`.
**Notes**: Docs only. Agent behavior unchanged.

## Entry 6 — Multi-turn run_agent_turn verification
**Goal**: Live 5-turn verification of `run_agent_turn` on an empty profile with accumulated history. Check whether `ask_targeted_follow_up` is selected naturally and whether the profile converges toward completeness. No application code changes.
**Plan proposed**: Scripted five user messages (name/email, education, skills, “what else do you need?”, then experience/project/achievement/target job). Print each turn’s tool, thought, parameters, observation, state_update, assistant_message, and full profile. No code changes except this log entry.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: `pip show google-genai` → 2.20.0. Key presence `SET` (value not printed). Ran five sequential `run_agent_turn` calls via project venv Python.
**Result**: Partial vs the two questions asked. Turns 1–3 and 5 selected `update_profile_field`. Turn 4 selected `inspect_completeness` (not `ask_targeted_follow_up`). `ask_targeted_follow_up` was never selected. Profile moved toward completeness (name, email, education, skills, one experience) but `inspect_completeness` at the end was `complete: false` with missing `contact.phone`, `projects`, `achievements`, `target_job`. `is_confirmed` stayed false. Exit 0.
**Notes**: One-tool-per-turn meant turn 5 stored only experience even though the user also stated a project, achievement, and target job. SDK AFC warning still printed. Application files not changed.

## Entry 7 — Multi-field update_profile_field
**Goal**: Stop silent data loss when one user message states multiple profile facts. Change `update_profile_field` to apply a `{field_name: value}` dict in one tool call. Do not change `ask_targeted_follow_up`.
**Plan proposed**: New signature `update_profile_field(profile, updates)`. Prompt requires all facts in one `{"updates": {...}}` payload. Still one tool per turn. `state_update` via existing `_profile_diff`. Re-run the exact Entry 6 five-message script. Isolated turn 4 uses a constructed post-turn-3 profile and **empty `conversation_history`** (simplified repro, not a full-history replay).
**Files changed**: `app/agent_engine.py`, `README.md` (Phase 2 note), `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/agent_engine.py`. Deterministic four-field write + `_profile_diff`. `pip show google-genai` → 2.20.0. Key `SET` (value not printed). Exact 5-turn live script. Isolated turn 4 with empty history.
**Result**: Pass for the data-loss fix. Deterministic `DETERMINISTIC_MULTI_FIELD=PASS`. Live turn 5 `updated_fields` = experience, projects, achievements, target_job; `state_update` has before/after for all four. Final completeness missing only `contact.phone`. In-script turn 4 and isolated turn 4 both selected `inspect_completeness`, not `ask_targeted_follow_up` (`ISO4 avoided_ask_targeted_follow_up: True`).
**Notes**: Isolated turn 4 is a simplified repro (empty history). Full-history replay is deferred to the `ask_targeted_follow_up` fix. `ask_targeted_follow_up` code was not changed. SDK AFC warning still printed.

## Entry 8 — ask_targeted_follow_up selection via system instruction
**Goal**: Get the LLM to choose `ask_targeted_follow_up` when the profile is incomplete and the user did not just supply facts. Prompt-only fix; no Python tool override; do not change the other four tools.
**Plan proposed**: Expand `_build_system_instruction` with inspect vs follow-up rules and a concrete example. Leave `ask_targeted_follow_up` body unchanged. Re-run the Entry 7 five-turn script and empty-history isolated Turn 4. Add a name-only two-follow-up scenario (`what next?` → fill email → `go on`).
**Files changed**: `app/agent_engine.py` (`_build_system_instruction` only), `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/agent_engine.py`. `pip show google-genai` → 2.20.0. Key `SET`. Exact 5-turn live script. Isolated Turn 4 with empty history. New name-only follow-up script; later turns hit 429 and were retried after waits, then stopped.
**Result**: Partial. Five-turn Turn 4: `ask_targeted_follow_up`, observation `What is your phone number?` (not `inspect_completeness`). Isolated Turn 4: same tool and question. New scenario FU1 (first attempt): LLM chose `ask_targeted_follow_up` asking for email. Consecutive follow-up after filling field #1 was not LLM-verified: FU2/FU3 and retries returned 429 `RESOURCE_EXHAUSTED` free-tier daily cap (20) for `gemini-3.6-flash`; fallback path selected `ask_targeted_follow_up`, which is not model choice. Model not switched.
**Notes**: `ask_targeted_follow_up` function body unchanged. Isolated Turn 4 remains an empty-history simplified repro; the 5-turn script is the full-history proof for Turn 4. SDK AFC warning still printed. Retry C when the daily quota resets.

## Entry 9 — Hard MAX_TURNS stop
**Goal**: Add a deterministic step/turn ceiling so an unconfirmed profile cannot loop forever. No Gemini API or tool-function changes.
**Plan proposed**: `MAX_TURNS = 15`. At the start of `run_agent_turn`, if `turn_number >= MAX_TURNS` and `not is_confirmed`, return `tool_name="step_limit_reached"` with `requires_input=False` and missing fields from `inspect_completeness`. No LLM call. Add `requires_input: bool = True` on `AgentTraceStep`. Verify with a mocked `call_gemini_json` loop; no live API.
**Files changed**: `app/agent_engine.py` (`MAX_TURNS`, `_step_limit_reached`, guard in `run_agent_turn`), `app/schemas.py` (`requires_input`), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/schemas.py app/agent_engine.py`. Monkeypatched `get_gemini_client` and `call_gemini_json`; ran turns 1..16 on an unconfirmed profile, then turn 15 on a confirmed profile.
**Result**: Pass. `STEP_LIMIT_TEST=PASS`. 14 mocked LLM turns; turn 15 and 16 returned `step_limit_reached` with `requires_input=False` and Gemini call count unchanged (14). Confirmed profile at turn 15 did call the mock (count 15). No live Gemini call.
**Notes**: Five tools and `_build_system_instruction` were not edited. `step_limit_reached` is not in `KNOWN_TOOLS`.

## Entry 10 — Assessment StudentProfile shape
**Goal**: Replace the simplified profile schema with the assessment shape (`target_roles` list, structured projects/experience, flat education fields) and update tool write/completeness/finalize logic to match.
**Plan proposed**: Rewrite `schemas.py` (drop `Contact`/`Education`; add `ProjectRecord`/`ExperienceRecord`). Update `UPDATABLE_FIELDS`, `_apply_one_field`, required checks, finalize (`target_roles` non-empty list), follow-up questions, and system-instruction tool descriptions. Do not change retrieve stub, `MAX_TURNS`, or `llm_client.py`. Skip live Gemini to conserve quota.
**Files changed**: `app/schemas.py`, `app/agent_engine.py` (tool constants, `_apply_one_field`, finalize, system-instruction field text), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/schemas.py app/agent_engine.py`. Direct tests: invented `student_profile_01`-shaped complete profile → `inspect_completeness` complete; finalize fails on empty `target_roles` and succeeds on a non-empty list; valid `projects` write; malformed project string rejected; links merge; retrieve stub unchanged.
**Result**: Pass. `SCHEMA_MIGRATION_TEST=PASS`. Completeness `missing_fields=[]`. Project reject: `projects[0] must be an object matching ProjectRecord`. No live `run_agent_turn` (quota conservation).
**Notes**: Complete-profile values were invented stand-ins; the uploaded eval file was not read. `document_id` defaults to `""` and is not tool-writable. `AgentTraceStep` unchanged. Retrieve stub and step-limit control flow unchanged.

## Entry 11 — Local embedding job retrieval
**Goal**: Replace the `retrieve_matching_jobs` stub with local MiniLM embeddings + cosine search over `data/jobs.jsonl`. Keep the unconfirmed-profile gate. No Gemini.
**Plan proposed**: Add `app/vector_store.py` (cache 9 job embeddings; query from roles/skills/experience/project text; keyword-sentence excerpts). Wire success path to `search_top_jobs`. Update only the retrieve tool paragraph in the system instruction. Add `sentence-transformers` to requirements. Skip live `run_agent_turn`. Do not read `student_profiles.jsonl`.
**Files changed**: `app/vector_store.py` (created), `app/agent_engine.py` (retrieve success path + retrieve tool description only), `requirements.txt`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: Installed `sentence-transformers==6.0.0` in the project venv. `python -m py_compile app/vector_store.py app/agent_engine.py`. Loaded `data/jobs.jsonl` (9 jobs, embeddings shape `(9, 384)`). `search_top_jobs` on three invented profiles (CV / NLP / data). `retrieve_matching_jobs` gate + success path.
**Result**: Pass. `RETRIEVAL_TEST=PASS`. CV top3: `computer_vision_engineer`, `trust_safety_ml_engineer`, `nlp_multimodal_ml_engineer` (top score 0.7378). NLP top3: `rag_agent_engineer`, `graduate_ai_engineer`, `agentic_ai_data_scientist` (0.7430). DATA top3: `graduate_ai_engineer`, `rag_agent_engineer`, `healthcare_ml_engineer`. Rankings differ. Unconfirmed still returns `profile must be confirmed first`. Confirmed returns `status=success` with 3 matches. No Gemini call. `student_profiles.jsonl` not read.
**Notes**: `job_type` and `location` had to be `str | None` because some job rows are null. Hugging Face printed an unauthenticated-request warning while downloading MiniLM; no token was set. Other four tools, step limit, and non-retrieve system-instruction text were not changed.

## Entry 12 — Gemini gap_analysis on matches
**Goal**: Add a per-match `gap_analysis` string from Gemini after ranking, without changing embeddings, ranking, or excerpts. Failures must fall back, not crash.
**Plan proposed**: `generate_gap_analysis` via `call_gemini_json` (`{"gap_analysis": str}`), using `build_query_text` plus job title/description. `get_gemini_client()` once per search; `None` or any API/JSON error → `Gap analysis unavailable for this match.` Stage A: compile + client-None fallback. Stage B: one CV `retrieve_matching_jobs` = 3 live calls (approved).
**Files changed**: `app/vector_store.py` (`generate_gap_analysis` + attach field after ranking), `app/agent_engine.py` (retrieve tool output shape only), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile`. Stage A monkeypatched `get_gemini_client` to `None` → all 3 matches had the fallback; top3 still `computer_vision_engineer`, `trust_safety_ml_engineer`, `nlp_multimodal_ml_engineer`. `pip show google-genai` → 2.20.0. Key `SET`. Stage B: one live `retrieve_matching_jobs` on the P1.3 CV profile (3 Gemini calls).
**Result**: Partial. Stage A pass. Stage B ranking unchanged (same top 3, scores 0.7378 / 0.5872 / 0.5428) and excerpts present, but all three `gap_analysis` values were the fallback string. Exceptions are swallowed by design, so the exact Gemini error was not printed. No 4th diagnostic call was made.
**Notes**: Ranking/excerpt code paths were not rewritten. AFC warning printed once during Stage B. Hugging Face unauthenticated warning ignored (model cached). Other tools and the step limit were not changed. A single follow-up diagnostic call would be needed to print the exact API/parse error.

## Entry 13 — Diagnose gap_analysis fallback
**Goal**: Log the real Gemini exception to stderr on gap-analysis failure, then spend one live `generate_gap_analysis` call to see why Stage B returned the fallback.
**Plan proposed**: Print `{type}: {exc}` to stderr in the existing except; also log empty/missing `gap_analysis` key. Return value unchanged. One live call vs `computer_vision_engineer` + the Stage B CV profile.
**Files changed**: `app/vector_store.py` (stderr log in `generate_gap_analysis` only), `docs/prompt_record.md` (this entry)
**Verification performed**: `pip show google-genai` → 2.20.0. Key `SET`. One `generate_gap_analysis` call.
**Result**: Partial (diagnosis succeeded). Function still returned `Gap analysis unavailable for this match.` Logged: `GeminiAPIError: Gemini generate_content failed: 429 RESOURCE_EXHAUSTED` — free-tier daily cap 20 for `gemini-3.6-flash` (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Retry hint was ~34s; quota id is daily, so a short wait is unlikely to restore calls.
**Notes**: Fallback string unchanged. Not a key-format / AQ. issue. SDK 2.20.0. No extra live calls.

## Entry 14 — Pending verification notes
**Goal**: Document items blocked on Gemini free-tier daily quota so the next session can resume without guessing.
**Plan proposed**: Docs only — append a PENDING section after Entry 13; update README build status. No application code.
**Files changed**: `docs/prompt_record.md` (this entry + PENDING section), `README.md` (build-phase status)
**Verification performed**: Read both files after the edit and confirmed the PENDING items and closed/pending/not-started lists are present.
**Result**: Pass. Documentation updated. No live Gemini call.
**Notes**: Do not attempt live Gemini until the user explicitly confirms quota has reset or billing is enabled.

## Entry 15 — Fact-bound CV markdown and A4 PDF
**Goal**: Add job-specific CV generation and A4 PDF export from confirmed profile facts only, with no new Gemini calls.
**Plan proposed**: `generate_cv_markdown(profile, job_match, job_description)` plus `generate_cv_pdf`. Matched skills via substring overlap. Reorder projects/experience by overlap with those skills. Growth Areas uses existing `gap_analysis` text if real, else deterministic fragments from a Requirements/Minimum qualifications segment. FPDF A4, 15mm margins, auto page break. No changes to agent_engine/vector_store/llm_client/schemas.
**Files changed**: `app/cv_generator.py` (created), `README.md`, `docs/prompt_record.md` (this entry). `fpdf2` was already in `requirements.txt`; installed `fpdf2==2.8.8` into the project venv.
**Verification performed**: `python -m py_compile app/cv_generator.py`. Invented long confirmed profile (4 projects, 3 experience). Two inline fake jobs (CV vs NLP). Compared markdown. Unconfirmed gate. Two A4 PDFs in a temp dir.
**Result**: Pass. `CV_TEST=PASS`. Matched skills A=`OpenCV, YOLO, video analytics, Docker` vs B=`NLP, RAG, LLM`. Lead project A=`Vision pipeline`, B=`RAG chatbot`. Unconfirmed raises `CvNotConfirmedError`. PDFs 4 pages each, page size ~210×297mm, files ~5.6KB.
**Notes**: No Gemini call. No `vector_store.search_top_jobs`. `job_description` is a separate argument. PENDING quota items 1–2 unchanged; P3.1 is no longer unstarted.

## Entry 16 — Close quota-blocked re-verifications
**Goal**: After the user confirmed quota reset, re-run Scenario C (3 turns) and one live `generate_gap_analysis`. Docs only; no application code.
**Plan proposed**: 4 live calls. Scenario C on a name-only Priya Sharma profile (`what next?` → supply the asked fact → `go on`). One `generate_gap_analysis` vs `computer_vision_engineer` + the Entry 11/12 CV profile. Log pass/fail honestly; remove both items from PENDING.
**Files changed**: `docs/prompt_record.md` (this entry + PENDING cleanup), `README.md` (those two items no longer pending)
**Verification performed**: `pip show google-genai` → 2.20.0. Key `SET`. Three `run_agent_turn` calls. One `generate_gap_analysis` call.
**Result**: Mixed.
- Scenario C: **Pass.** FU1 `ask_targeted_follow_up` asked for `degree` (`What degree or qualification did you complete?`). FU2 wrote `degree=Bachelor of Computing Science`. FU3 `ask_targeted_follow_up` asked for `institution` (`Which institution did you study at?`) — a different field, not a repeat. Neither follow-up was a FALLBACK.
- `generate_gap_analysis`: **Fail for output quality.** Not a 429. Stderr: `gap_analysis failure: empty or missing gap_analysis key (got NoneType)`. Return value stayed `Gap analysis unavailable for this match.` No grounded prose was produced. The model call succeeded but the JSON object did not contain `gap_analysis`.
**Notes**: Current schema has no email field, so Turn 2 supplied degree (what FU1 asked), not email. 4 live calls used. No application code changed.

## Entry 17 — Diagnose gap_analysis JSON keys
**Goal**: Log the full parsed Gemini object when `gap_analysis` is missing, then one live call to see the actual keys.
**Plan proposed**: Add `raw_parsed=...` to the existing stderr line. Do not change the fallback return. One live call vs `computer_vision_engineer` + CV profile. Propose a fix; do not implement it yet.
**Files changed**: `app/vector_store.py` (stderr log of full parsed dict only), `docs/prompt_record.md` (this entry)
**Verification performed**: `pip show google-genai` → 2.20.0. Key `SET`. One `generate_gap_analysis` call.
**Result**: Pass as diagnosis. Fallback still returned. Raw parsed JSON was `{"feedback": "The student demonstrates strong alignment with core computer vision requirements, including PyTorch, OpenCV, video analytics, object detection, and YOLO. However, their profile lacks the required 3 to 5 years of professional experience as well as documented experience with model deployment, cloud platforms, and large datasets."}`. The model used `feedback` instead of `gap_analysis`. Content looks grounded (3–5 years, OpenCV, YOLO, deployment, cloud, large datasets are in that job text) but is discarded because of the key name.
**Notes**: Fix not applied. Proposed options: tighten the system instruction to require the exact key, or enforce `response_schema` on `generate_content`. Awaiting user choice.

## Entry 18 — Enforce gap_analysis via response_schema
**Goal**: Guarantee the Gemini JSON key is `gap_analysis` by passing a structured `response_schema` through `call_gemini_json`, without changing the fallback return string.
**Plan proposed**: Add optional `response_schema: dict | None = None` to `call_gemini_json` and pass it to `GenerateContentConfig` only when set. In `generate_gap_analysis`, send a Gemini Schema requiring `gap_analysis` (string). Existing callers unchanged. One live call vs `computer_vision_engineer` + the CV profile.
**Files changed**: `app/llm_client.py` (optional `response_schema` on `call_gemini_json`), `app/vector_store.py` (`_GAP_RESPONSE_SCHEMA` passed into the call), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/llm_client.py app/vector_store.py`. `pip show google-genai` → 2.20.0. Key `SET`. One `generate_gap_analysis` call; wrapped `call_gemini_json` to inspect the parsed object without a second API call.
**Result**: Pass. Parsed keys were `['gap_analysis']` (not `feedback`). Returned text was not the fallback: "The student demonstrates strong alignment with core technical requirements including PyTorch, OpenCV, object detection, and video analytics. However, they lack the required 3 to 5 or more years of professional experience in AI/computer vision, as well as explicit experience in CV model deployment, image classification, and working with large datasets." Grounded in the real job (3–5 years, OpenCV, PyTorch, object detection, video analytics, model deployment, image classification, large datasets).
**Notes**: Fallback string and except-path unchanged. Temporary `raw_parsed` stderr log left in place. Agent `run_agent_turn` still calls `call_gemini_json` without a schema. SDK AFC warning printed once. 1 live call used.

## Entry 19 — Streamlit app shell
**Goal**: Wire the existing agent, retrieval, and CV functions into a Streamlit UI without changing those modules. Hide chat once `job_matches` is populated so a second retrieval cannot fire.
**Plan proposed**: Replace `streamlit_app.py`. Split CV markdown on `## ` headings; preamble is Summary; map Growth Areas back to `Missing Requirements / Growth Areas`; keep Education. Widget-key text areas + reassemble preview. Reuse agent retrieve matches when present, else one `search_top_jobs`. Stage A = 0 Gemini. Stage B live conversation left pending user budget approval (5–7 calls).
**Files changed**: `streamlit_app.py` (replaced), `README.md`, `docs/prompt_record.md` (this entry). Installed `streamlit==1.62.0` into the project venv (`requirements.txt` already listed `streamlit`).
**Verification performed**: `python -m py_compile streamlit_app.py`. Split/reassemble unit check on `generate_cv_markdown` output. Streamlit `AppTest`: empty completeness + chat input; seeded `job_matches` hides chat input; Start New Profile restores chat. Seeded confirmed CV profile + job card: Generate CV, edit Matched Skills shows `(edited)`, Reset restores, Regenerate PDF produces download bytes. Live server started at `http://localhost:8501`. No Gemini calls.
**Result**: Pass for Stage A (and a 0-call CV editor path). Chat lock works. Frozen app modules were not edited.
**Notes**: Browser MCP tools were unavailable; UI was verified with Streamlit AppTest plus a running local server. Stage B (conversation → confirm → live `search_top_jobs`) not run. Activities section is omitted when empty, as planned.

## Entry 20 — Demo cards and onboarding
**Goal**: Add a deterministic welcome message, five JSONL demo profile cards, and a blank-profile start, without changing agent_engine logic.
**Plan proposed**: Cards at the top of the main column. Button sets `pending_demo_action` then a top-of-script handler runs. 01/02/05 load `student_profiles.jsonl`, set `is_confirmed=True`, wrap `get_gemini_client` to `None` around `search_top_jobs` (0 Gemini). 03/04 feed the intro into `run_agent_turn` (not live-tested). Blank resets. Onboarding text from `inspect_completeness` on an empty profile, not stored in history.
**Files changed**: `streamlit_app.py`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile streamlit_app.py`. AppTest: onboarding sentence present; six demo buttons; Maya Chen skip confirms, 3 job cards, chat locked, skip note shown, `call_gemini_json` count 0; Start New Profile and blank card restore an empty session. No live 03/04 call.
**Result**: Pass for the 0-call paths. 03/04 live follow-up not run.
**Notes**: The skip-path Gemini wrap is a testing convenience; remove it later for one real run per 01/02/05 card (~9 gap_analysis calls) before graded evidence. `student_profile_02` has empty `experience` in the JSONL, so sidebar completeness can still list that field after skip. Extra JSONL keys are ignored by `StudentProfile.model_validate`.

## Entry 21 — Skip-path load time and PDF wrap
**Goal**: Diagnose slow 01/02/05 skip-path loading and PDF text cut off at the right margin; fix both without Gemini.
**Plan proposed**: Time model load vs embed vs search before assuming Streamlit remounts MiniLM. Fix PDF if `multi_cell` leaves the cursor at the right edge. Warm retrieval via `@st.cache_resource` so the skip click does not pay first-load cost.
**Files changed**: `app/vector_store.py` (load timing), `streamlit_app.py` (`@st.cache_resource` bundle + warm on startup), `app/cv_generator.py` (`_write_wrapped` with `new_x=LMARGIN`), `docs/prompt_record.md` (this entry)
**Verification performed**: AppTest timing before/after. Ethan Nguyen PDF x-trace of every `multi_cell`. P3.1 long Jordan Lee profile, two fake jobs, 4-page A4 PDFs. No Gemini.
**Result**: Pass.
- Issue A: Not a per-rerun reload. Module singleton already survived the second skip (13.11s then 0.054s). Slowness was first MiniLM load (~13s) happening *on the demo click*. After warming at startup: app start 14.9s, skip 01 0.092s, skip 05 0.060s.
- Issue B: Confirmed. fpdf2 `multi_cell` defaults to `new_x=RIGHT`, so the 2nd+ bullet started at x=195, 375, … (linkedin, Connected…, Fixed s…). After the fix every line starts at x=15. Ethan 1 page; Jordan still 4 pages, ~210×297mm.
**Notes**: No page-image renderer was installed (`pypdf`/`fitz` missing); wrap was verified by cursor x and full strings still present in the markdown fed to the PDF.

## Entry 22 — Live demo card 03 (1 Gemini call)
**Goal**: Run the conversational demo path for `student_profile_03` and report the real turn-1 tool and message.
**Plan proposed**: Click the Sofia Williams card via Streamlit AppTest (same handler as the UI; browser MCP unavailable). Count `call_gemini_json`. Expect the JSONL intro as the first user message and a targeted follow-up, likely `target_roles`.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: `pip show google-genai` → 2.20.0. Key `SET`. One AppTest click of `demo_student_profile_03`.
**Result**: Path wiring **pass**; follow-up content **did not match** `profile_status`. Exactly **1** Gemini call. Turn 1 used `update_profile_field`, not `ask_targeted_follow_up`. First user message was the exact intro. `is_confirmed=False`, `job_matches=[]` (not the 01/02/05 skip). Assistant said it had already written degree/year/projects **and** target roles. It stored `target_roles=['Data Roles', 'Software Roles']` from “still deciding whether to apply for data or software roles” instead of asking which role. Still missing name, technical_skills, experience. Chat remains open.
**Notes**: `profile_status` predicted a target-role follow-up. The model treated the indecision as two role labels. No second call.

## Entry 23 — Do not write undecided fields
**Goal**: Stop the agent writing guessed list values when the user is still deciding. Prompt-only change in `_build_system_instruction`.
**Plan proposed**: Add an uncertainty rule and a concrete `target_roles` example. Keep Entry 8 “what next?” → `ask_targeted_follow_up` and Entry 7 multi-field writes for *committed* facts. One live 03 card click.
**Files changed**: `app/agent_engine.py` (`_build_system_instruction` only), `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile app/agent_engine.py`. Read the new instruction against Entries 6–8/16. One AppTest click of `demo_student_profile_03`. Exactly 1 Gemini call. No 04 live call.
**Result**: Pass for the visible 03 path. `tool_name=ask_targeted_follow_up`. `target_roles` stayed `[]`. Assistant: “Would you like to target both Data and Software roles, or pick one for now?” Matches `requires a target-role follow-up`. Observation from Python was “What is your full name?” because `missing_fields` listed every empty required field (name first); the UI shows `assistant_message`, which asked about roles.
**Notes**: Instruction still says vague “what next? / go on” uses `ask_targeted_follow_up`, inspect stays silent-only, and committed multi-field updates remain required. Profile 04’s “have not yet decided” is named as a trigger phrase. No tool function bodies changed.

## Entry 24 — Diagnose assistant_message vs follow-up observation
**Goal**: Explain why turn-1 Agent Trace showed a name question in `observation` while the user saw a target-roles question. Read-only; no code change, no live call.
**Plan proposed**: Read `run_agent_turn` dispatch and Streamlit history append.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: Read `run_agent_turn`, `_dispatch_tool`, `_fallback_turn`, and `streamlit_app.py` history write.
**Result**: The LLM's `assistant_message` is what the user sees. On the success path it is copied from the model JSON and never replaced by `ask_targeted_follow_up`'s `{"question": ...}`. That tool result is stored only in `observation`. Fallback turns *do* use the tool question when `assistant_message` is empty. Streamlit appends `step.assistant_message`, not `observation["question"]`.
**Notes**: Trace `observation` is a faithful record of the tool, not of the visible chat line, on a successful `ask_targeted_follow_up` turn. Closed as an understood design decision: the canned tool question is a fallback-path safety net; `assistant_message` is the primary UX.

## Entry 25 — Chat fragment so the page does not grey out
**Goal**: Stop Streamlit's full-page rerun overlay on every chat submit. Refresh sidebar completeness and Agent Trace from the same fragment. Leave Start New Profile outside the fragment.
**Plan proposed**: `@st.fragment` on the chat renderer (confirmed on Streamlit 1.62.0). `st.spinner("Thinking...")` around `_handle_user_message`. No app-level `st.rerun()` after a normal turn; full `st.rerun()` only when `job_matches` first becomes non-empty. Mocked `run_agent_turn` for verification; no Gemini.
**Files changed**: `streamlit_app.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `hasattr(st, "fragment")` True on 1.62.0. `python -m py_compile streamlit_app.py`. AppTest with monkeypatched `run_agent_turn`: one chat submit updated history, turn 1, profile name, and agent_trace; Start New Profile still present; 0 Gemini calls. AppTest's sidebar element tree still showed the pre-turn captions after the fragment path (likely a test-harness limit). No browser MCP to watch the overlay.
**Result**: Pass for wiring and mocked chat state. Visual “only chat greys out” needs a look in the running app.
**Notes**: Demo-card clicks still full-script rerun. `agent_engine.py` and `llm_client.py` unchanged.

## Entry 26 — Diagnose chat lag and premature confirm
**Goal**: Root-cause two live bugs without changing code: chat lagging one turn after the fragment change, and confirmation with only `target_roles` filled.
**Plan proposed**: Read `_render_chat_fragment` order and `finalize_and_confirm_profile` vs the system instruction. No live call.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: Read `streamlit_app.py` fragment and `app/agent_engine.py` finalize + `_build_system_instruction`.
**Result**: Diagnosis only; no fix.
**Notes**: See the two findings in this conversation. Awaiting approval before changing fragment rerun or confirm guidance.

## Entry 27 — Fragment repaint and complete-profile confirm gate
**Goal**: Repaint the chat fragment after each turn, and reject `finalize_and_confirm_profile` until every required field is filled.
**Plan proposed**: `st.rerun(scope="fragment")` after a normal chat turn. Python gate: empty `target_roles` → existing error; roles set but incomplete → `profile incomplete` + `missing_fields`; else confirm. Instruction text updated to match. Deterministic tests first; live “target role only” path not run until the call budget is approved.
**Files changed**: `streamlit_app.py` (fragment rerun), `app/agent_engine.py` (`finalize_and_confirm_profile` + instruction), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `python -m py_compile`. Roles-only profile → `profile incomplete` with missing name/degree/institution/year/skills/projects/experience. Empty roles → `target_roles required`. Invented complete profile → success. JSONL: 01 and 05 finalize successfully; 02 fails on empty `experience` (skip path still sets `is_confirmed=True` without this gate). Live: 2 `run_agent_turn` calls on a blank profile (`pip show google-genai` 2.20.0, key `SET`).
**Result**: Pass. Deterministic gate as above. Live: Turn 1 `update_profile_field` wrote `target_roles=['Software Engineering Intern']`, `is_confirmed=False`, remaining required fields still missing. Turn 2 user “that is all I have for now, go on” → `ask_targeted_follow_up` asking for name, not `finalize_and_confirm_profile`. Still unconfirmed. Exactly 2 Gemini calls.
**Notes**: Scenario C is prompt-only follow-up selection and does not call finalize; instruction still has the “what next? / go on” example. Demo 01/02/05 skip path does not call `finalize_and_confirm_profile`.

## Entry 28 — JSONL required-field audit; Bug 1 visual not driven from here
**Goal**: List empty required fields on all five `student_profiles.jsonl` rows. Try to confirm the chat fragment repaint in the running UI.
**Plan proposed**: `inspect_completeness` on each JSONL profile. Browser MCP unavailable; AppTest chat submit is a full-script run so `st.rerun(scope="fragment")` raises there (not proof of live overlay).
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: Loaded 01–05 through `StudentProfile.model_validate` + `inspect_completeness`.
**Result**: Only `student_profile_02` (Liam Patel) is incomplete: `experience=[]`. 01, 03, 04, 05 have all required fields filled. Bug 1 overlay/immediate reply was not visually confirmed in this session (no browser tools).
**Notes**: Empty `experience` on 02 is a one-off in this dataset, not a pattern.

## Entry 29 — CV redesign and adaptive section order
**Goal**: Restructure the exported CV (markdown + PDF + Streamlit editor) without changing job-card ranking, gap analysis, or frozen modules. Add deterministic Projects vs Experience ordering from overlap strength.
**Plan proposed**: Drop Target Roles and Growth Areas from the document. Rename Matched Skills to Skills; broaden candidates to `technical_skills` + project technologies + description tokens (len≥4, stoplist) that appear in the job text; render as one comma-separated line. Keep within-section reorder on that list. PDF accent RGB(27,79,114) on name and `##` headers with a 0.4mm rule; bold titles; keep `_write_wrapped` `new_x=LMARGIN`. Adaptive order: Education, Skills, then the stronger of Projects/Experience (`_overlap_count > 0` entry counts; ties → Projects first), Activities last. Editor heading set and `cv_section_order` preserve that order on reassemble.
**Files changed**: `app/cv_generator.py`, `streamlit_app.py`, `README.md`, `docs/prompt_record.md` (this entry). Not changed: `app/agent_engine.py`, `app/vector_store.py` ranking, `app/schemas.py`.
**Verification performed**: `python -m py_compile`. Maya Chen (`student_profile_01`) vs cosine top-1 with `get_gemini_client` wrapped to `None`. Jordan Lee P3.1 two fake jobs (vision vs NLP). Streamlit source grep + AppTest editor labels. 0 Gemini calls.
**Result**: Pass for the requested checks. Maya: no Target Roles/Growth Areas; Skills one line; PDF accent + left-margin wrap; Projects before Experience (strength 2 vs 1). Jordan: lead project Vision pipeline vs RAG chatbot; skills differ; section *order* did not differ (ties keep Projects first). Editor labels: Summary, Education, Skills, Projects, Experience, Activities (or adaptive body order); no leftover Growth Areas / Target Roles / Matched Skills.
**Notes**: Maya top job was `agentic_ai_engineer` (score 0.5942). Broadened Skills can include generic description tokens that also appear in the job (`design`, `agent`, `tool`, `team`). Job cards unchanged.

## Entry 30 — Filter generic description-derived skills
**Goal**: Stop lowercase/generic English words from description text appearing on the Skills line when they overlap job prose.
**Plan proposed**: Description-mined tokens only. Require proper-noun / acronym shape (an uppercase letter, or internal `.` / digit / `+#`). Small denylist for Title-Case English (`team`, `document`, `design`, …). Curated `technical_skills` and `project.technologies` stay unfiltered. Strip trailing sentence punctuation so `tests.` is not treated as a tech token.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Maya Chen vs the same cosine top job `agentic_ai_engineer` with Gemini wrapped off. 0 Gemini calls.
**Result**: Pass. Skills line is `Python` only. No `design` / `agent` / `tool` / `team`. Description-mined keepers from her projects: `Python`, `REST` (`REST` is not in that job text so it stays off the Skills line).
**Notes**: Capitalization is the main gate because the observed leaks were all-lowercase. The denylist is a safety net for Title-Case English, not the primary filter.

## Entry 31 — Skills lists the full genuine set
**Goal**: Skills should represent what the candidate knows, not the subset that happens to appear in the selected job description.
**Plan proposed**: `_cv_skills` returns `technical_skills` + project technologies + capitalized description terms, with no job-text filter. Job overlap (`_skills_mentioned_in_job`) is used only for Projects/Experience reorder and adaptive section strength.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Maya Chen vs `agentic_ai_engineer` (same cosine top job), Gemini wrapped off. 0 Gemini calls.
**Result**: Pass. Skills line is `Python, Java, SQL, REST APIs, Git, GitHub, pytest, Jupyter Notebook, Flask, SQLite, REST, prompt design, tool calling, JSON`. Expected techs (Python, Flask, SQLite, REST APIs, pytest, Git) are present. No `design` / `agent` / `tool` / `team`. Job-overlap used for reorder is `['Python']`; lead project is Team Course Planner (both projects mention Python; original order breaks the tie).
**Notes**: `prompt design` and `tool calling` come from `projects[].technologies` in the JSONL, not from description mining.

## Entry 32 — Grouped follow-ups and one-time optional activities prompt
**Goal**: Ask name/degree/institution/year in one follow-up; after required fields are complete, offer one optional certifications/awards/activities question that cannot block confirmation.
**Plan proposed**: `GROUPED_FIELDS` vs `INDIVIDUAL_FIELDS` in `ask_targeted_follow_up`. Track the optional prompt via the exact canned `ACTIVITIES_PROMPT` string in conversation history (no schema flag). Inject Python-computed follow-up state into `_build_system_instruction`. Do not change `update_profile_field`, `inspect_completeness`, `finalize_and_confirm_profile`, `retrieve_matching_jobs`, or the step limit.
**Files changed**: `app/agent_engine.py`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: Deterministic grouping/status checks (0 Gemini). Then 4 live `run_agent_turn` calls (`google-genai` 2.20.0, key SET, default `gemini-3.6-flash`).
**Result**: Pass. Call 1 empty profile → `ask_targeted_follow_up` with `["name", "degree", "institution", "year"]`. Call 2 one message wrote all four (`Maya Chen` / `Bachelor of Computing Science` / `University of Technology Sydney` / `3rd year`). Call 3 complete required fields, empty activities → `tool_name=none` and the exact optional prompt; not confirmed. Call 4 `"none"` → `finalize_and_confirm_profile` success, `is_confirmed=True`, `activities` still `[]`.
**Notes**: Turn 1 `assistant_message` paraphrased the grouped question (added a welcome) while tool parameters and the canned observation question were the grouped set. Exactly 4 Gemini calls.

## Entry 33 — Activities prompt on the last-required-field transition
**Goal**: Check whether the optional activities question fires on the realistic path: last required field is written, then the next user turn — not from a pre-seeded complete profile.
**Plan proposed**: Python-seed all required fields except `target_roles` plus a short mid-conversation history. Two live `run_agent_turn` calls. No application code changes.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 2 live calls (`google-genai` 2.20.0, key SET). Pre-state `missing_fields=['target_roles']`.
**Result**: Pass for the next-turn ask. Transition turn (user: `"Software Engineering Intern."`) → `update_profile_field` wrote `target_roles`; `assistant_message` only acknowledged the role; did **not** ask activities. Next turn (user: `"go on"`) → `tool_name=none`, exact `ACTIVITIES_PROMPT`, still unconfirmed, `activities=[]`.
**Notes**: Exactly 2 Gemini calls. The optional prompt does not share the last `update_profile_field` turn (one tool per turn), which is the intended sequence.

## Entry 34 — Adaptive CV layout, skill categories, grounded summary
**Goal**: Two-column layout when richness ≥ 8, categorized Skills, and a `response_schema` Gemini professional summary with a grounding check and templated fallback.
**Plan proposed**: `cv_richness` = skills + projects + experience. Sidebar Contact/Education/Skills vs main Summary/Projects/Experience/Activities. Fixed skill alias table with Other Skills catch-all. `generate_cv_summary` uses `{"summary": str}` schema; ungrounded or API failure → template; stderr logs `path=gemini` or `path=fallback`.
**Files changed**: `app/cv_generator.py`, `streamlit_app.py`, `README.md`, `docs/prompt_record.md` (this entry). Not changed: `app/agent_engine.py`, `app/vector_store.py` ranking, `app/schemas.py`.
**Verification performed**: 0-call layout/category/grounding/PDF/Jordan reorder checks. One live `generate_cv_summary` for Maya Chen vs cosine top job `agentic_ai_engineer` (search Gemini wrapped off).
**Result**: 0-call checks pass (Maya richness 11 → two-column; sparse 4 → single; categories keep every skill; fabricated Kubernetes/led-development summary fails grounding). Live call **1/1**: `cv_summary path=fallback reason=ungrounded ['Third-year Bachelor', 'Third-year']` — Gemini returned a grounded-looking summary that hyphenated the profile's `Third year`. CV showed the templated summary. Treated as a valid safeguard result, not a test failure. Hyphen/token grounding was then tightened so `Third-year` matches `Third year`; Kubernetes still fails. No second live call.
**Notes**: Job-card ranking unchanged. Streamlit job select now calls `generate_cv_summary` when a Gemini client exists (1 call per generate).

## Entry 35 — Retry Maya professional summary after hyphen grounding fix
**Goal**: One live `generate_cv_summary` for Maya Chen vs `agentic_ai_engineer` after hyphen/space normalization.
**Plan proposed**: Load JSONL profile + that job description; do not call `search_top_jobs`. Exactly one `call_gemini_json`.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 1 live call (`google-genai` available, key SET).
**Result**: `path=fallback reason=api GeminiAPIError: 503 UNAVAILABLE` (model high demand). Shown text is the template: "Maya Chen is a Third year Bachelor of Computing Science student at University of Technology Sydney with experience in Python, Java, SQL, seeking Software Engineering Intern." Grounding of that template is clean. The hyphen fix was not exercised because no summary JSON was returned.
**Notes**: Exactly 1 Gemini call. No retry (would be a second call).

## Entry 36 — Second retry of Maya generate_cv_summary
**Goal**: One more live `generate_cv_summary` for Maya Chen vs `agentic_ai_engineer`.
**Plan proposed**: Same JSONL load as Entry 35. Exactly one call.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 1 live call. Key SET.
**Result**: `path=fallback reason=ungrounded ['PyTest. Her']`. CV text is the template: "Maya Chen is a Third year Bachelor of Computing Science student at University of Technology Sydney with experience in Python, Java, SQL, seeking Software Engineering Intern." API succeeded this time; grounding rejected a title-case span that glued `PyTest.` to the next sentence's `Her`.
**Notes**: Exactly 1 Gemini call. No retry.

## Entry 37 — Sentence-boundary fix for summary grounding
**Goal**: Stop claim extraction from merging a term at the end of one sentence with the first word of the next (`PyTest. Her`).
**Plan proposed**: Split the summary on `.!?` + whitespace + capital letter, then extract title-case phrases and tech tokens per sentence. Multi-word phrases no longer allow `.` inside a token.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Hardcoded strings only. 0 Gemini calls.
**Result**: Pass. `"...PyTest. Her projects..."` splits into two sentences; claims include `pytest` and not `PyTest. Her`; grounding of that text is clean. Fabricated Kubernetes / `led development` still fail. A fully profile-true summary still passes.
**Notes**: `PyTest` normalizes to the profile skill `pytest`, so it is recorded under that form rather than the mixed-case spelling.

## Entry 38 — Retry Maya generate_cv_summary after sentence-boundary fix
**Goal**: One live `generate_cv_summary` for Maya Chen vs `agentic_ai_engineer`.
**Plan proposed**: Same JSONL load. Exactly one call.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 1 live call. Key SET.
**Result**: `path=fallback reason=ungrounded ['Skilled']`. CV text is the template: "Maya Chen is a Third year Bachelor of Computing Science student at University of Technology Sydney with experience in Python, Java, SQL, seeking Software Engineering Intern."
**Notes**: Exactly 1 Gemini call. Title-case English adjective `Skilled` was treated as a claim. No retry.

## Entry 39 — Skip sentence-initial English in summary grounding
**Goal**: Stop flagging ordinary first words (`Skilled`, `With`) as unverifiable claims while still catching invented tech terms.
**Plan proposed**: The first word of a sentence is not a standalone claim unless it is a known skill. A sentence-initial capitalized phrase is kept only if the full phrase is in the profile, or the first word is a known skill / all-caps / camelCase (`REST APIs`, `Team Course Planner`). `With Docker` is dropped as a phrase; `Docker`/`Kubernetes` still extract later in the sentence.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Hardcoded Maya strings. 0 Gemini calls.
**Result**: Pass. (1) `Skilled in Python and SQL...` does not flag Skilled; Python/SQL grounded. (2) Kubernetes / led development still fail. (3) `PyTest. Her` still splits and passes. (4) `With Docker and Kubernetes...` does not flag With; Kubernetes (and Docker) still fail.
**Notes**: Known-term check for the first word uses the skill list, not the full profile blob, so words like `with` in an experience sentence cannot whitelist `With`.

## Entry 40 — Retry Maya generate_cv_summary after sentence-initial fix
**Goal**: One live `generate_cv_summary` for Maya Chen vs `agentic_ai_engineer`.
**Plan proposed**: Same JSONL load. Exactly one call.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 1 live call. Key SET.
**Result**: `path=gemini`. Summary: "Third-year Bachelor of Computing Science student at the University of Technology Sydney with experience in Python, REST APIs, SQL, and pytest. Created a tool-call agent prototype in Python that selects local course tools and records results, alongside building the Python service layer for a course planner web API. Seeking to apply prompt design, tool-calling capabilities, and backend development skills to build and deploy software solutions."
**Notes**: Exactly 1 Gemini call. Not the template.

## Entry 41 — Fix ImportError + auto PDF on job select
**Goal**: Unblock Streamlit start (`generate_cv_summary` import) and show Download immediately after job select, without requiring a manual regenerate unless sections were edited.
**Plan proposed**: Diagnose first: `generate_cv_summary` is defined in `app/cv_generator.py` (not a leftover snippet). Auto-write PDF in `_select_job`. Track dirty state vs `cv_pdf_sections` (last PDF snapshot), not `cv_sections_original`, so Reset still works and regenerate is only needed after edits since the last PDF write.
**Files changed**: `streamlit_app.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: AppTest with Gemini wrapped off (`get_gemini_client` → None). 0 Gemini calls.
**Result**: Pass. Import succeeds. Job select yields 2417-byte PDF and Download immediately; Regenerate hidden until Skills edited; download bytes unchanged until Regenerate; after Regenerate, PDF changes and the button hides.
**Notes**: ImportError was not a missing definition — the function is in `cv_generator.py` line 313. A stale Streamlit process that imported the module before that commit would still raise it; restart the app after pull/save.

## Entry 42 — Manual CV section-order override (single-column)
**Goal**: Let the user reorder Skills / Professional Summary / Projects / Experience / Activities in single-column layout, layered on the automatic adaptive default. Two-column stays fixed. Header/Education stay first. Job switch resets to that job's automatic order. Order changes are dirty until Regenerated PDF.
**Plan proposed**: One `st.selectbox` per present reorderable section (1st–Nth). Changing a dropdown moves that section to the chosen rank and shifts the others, so positions cannot collide. Source of truth is `cv_section_order`; `cv_pdf_order` is the last-PDF snapshot used with `cv_pdf_sections` for dirty tracking. Two-column renders the same dropdowns `disabled=True` with the fixed-layout caption.
**Files changed**: `streamlit_app.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: AppTest with Gemini wrapped off. Maya (richness 11, two-column) and a constructed richness-4 profile with empty activities. 0 Gemini calls.
**Result**: Pass. Two-column: five dropdowns disabled + caption. Single-column: four enabled dropdowns pre-filled with automatic order (Activities omitted). Moving Experience to 1st dirties the PDF without changing bytes; Regenerated updates preview/PDF to Education → Experience → Skills → Professional Summary → Projects. Switching jobs restores the new job's automatic order.
**Notes**: Regenerated now reruns only after the section text areas have rendered, so Streamlit does not drop `cvedit_*` state (that was wiping the live preview).

## Entry 43 — Swallow OSError from CV diagnostic stderr prints
**Goal**: Stop `print(..., file=sys.stderr)` in `generate_cv_summary` from crashing job select on Windows (`OSError: [Errno 22] Invalid argument`).
**Plan proposed**: One `_log_cv_diag` helper wrapping every diagnostic stderr print in `cv_generator.py`. Catch `OSError` only and ignore it. Logging must never abort CV generation.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Forced `sys.stderr.write` to raise `OSError(22)` and exercised all four summary paths (api / empty / ungrounded / gemini). AppTest job select with Gemini wrapped off. 0 Gemini calls.
**Result**: Pass. Broken-stderr prints do not raise; fallback and gemini paths still return the expected summary. Job select produces a PDF and a Professional Summary (`Maya Chen is a Third year...`).
**Notes**: If an already-running Streamlit process was started before this save, trigger a rerun or restart so it reimports `cv_generator`.

## Entry 44 — Three CV rendering/extraction fixes from Ethan Nguyen's CV
**Goal**: (1) Contact lines showed literal `* github: ...`. (2) Words split mid-character in the sidebar (`UT` / `S`). (3) Skills "Other Skills" contained the verbs Worked / Implemented / Connected / Fixed.
**Plan proposed**: (1) Emit Contact as plain `Label: value` lines with cased labels (`GitHub`, `LinkedIn`), no bullet — cleaner than a glyph for 1-2 short links. (2) `_write_wrapped` picked `wrapmode` by comparing the whole string to the column width, so every paragraph longer than one line got CHAR wrapping; choose CHAR only when a single token cannot fit (`_wrap_mode`). (3) Give `_description_terms` the sentence-initial rule already used by the summary grounding checker.
**Diagnosis correction**: the mid-word bug was not sidebar-specific and there was no second wrapping path — both columns share `_write_wrapped`. Measured: Ethan's education line is 197.8mm against the 62mm sidebar and his summary 249.2mm against the 111mm main column, so both took the CHAR branch. The screenshot shows it in the main column too (`Python, J` / `ava`, `soft` / `ware`, `demons` / `tration`); the wider column just hides it better.
**Files changed**: `app/cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: Regenerated Ethan (`student_profile_05`) and Maya (`student_profile_01`) against `agentic_ai_engineer`, capturing every text run's real line breaks via `multi_cell(dry_run=True, output="LINES")` and checking each junction lands on whitespace in the source. 0 Gemini calls (templated summary).
**Result**: Pass. Ethan Contact: `GitHub: github.com/example/student-profile-05`, `LinkedIn: linkedin.com/in/example-student-05`, no asterisks. 31 text runs, 0 CHAR-mode runs, 0 mid-word breaks in either column, both profiles. Other Skills now `CI basics, unit testing, Jira-style task tracking, structured tool calls, JSON`; the fix dropped exactly `Worked, Implemented, Connected, Fixed`. Maya's skills are unchanged (0 dropped) and her summary-grounding cases still behave: Skilled/With not flagged, Kubernetes and `led development` still flagged, `PyTest. Her` still clean.
**Notes**: Two follow-on details. `Implemented Java backend endpoints` first survived because the "starts a capitalized phrase" allowance accepted verb + proper noun; a sentence-initial phrase now counts only when the profile lists it verbatim (`Machine Learning` yes, `Implemented Java` no), and `Java` is still mined mid-sentence. Contact URLs are displayed without `https://`/`www.` so a link fits the 62mm sidebar as one token (60.0mm and 55.5mm) and wraps after the label instead of mid-URL; that was the only remaining CHAR-mode text. Bullets in Projects/Experience/Activities still render as `* ` by choice — out of scope for this fix.

## Entry 45 — Elaboration follow-ups for thin project/experience descriptions
**Goal**: When a field has content but is too thin for a CV, ask for detail instead of silently accepting it.
**Plan proposed**: `detect_thin_content(profile)` flags any project/experience description under 8 words (pure Python). A 6th tool `ask_elaboration_question(item_type, item_title)` returns the fixed template question. `_build_system_instruction` publishes `thin_content` and `next_elaboration`, with the priority resolved in Python: `next_elaboration` is non-null only when `required_complete` is true, the entry is thin, and it has not already been asked about. Tracking mirrors `_activities_prompt_already_asked` — an assistant turn containing the fixed question tail plus that entry's title counts as asked.
**Write-path finding**: `_apply_one_field` only supported whole-list replacement for `projects` / `experience`, so writing one elaborated entry would have destroyed the siblings. Approved option A: two new index-addressed field names, `project_description` and `experience_description`, each taking `{"index": int, "description": str}` and handled by a new `_apply_item_description`. `update_profile_field`'s body and every pre-existing field branch are unchanged; whole-list writes still behave exactly as before. `inspect_completeness`, `finalize_and_confirm_profile`, `retrieve_matching_jobs` and `MAX_TURNS` untouched. No Streamlit change needed — the trace panel renders `tool_name` generically.
**Files changed**: `app/agent_engine.py`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: 39 deterministic checks, 0 Gemini calls, all passing: thin detection (7 words thin, exactly 8 not thin, empty thin, blank title falls back to `unnamed`, substantive never flagged, correct type/index/title); the exact question string and its dispatch; index-addressed writes (index 1 of a 3-project list replaced, siblings byte-identical, target's title/context/technologies intact, no other profile field changed, entry no longer thin, `experience_description` equivalent); seven rejection cases (out-of-range, negative, non-dict, missing key, string index, bool index, non-string description) each erroring with zero mutation; whole-list `projects` regression; `_next_elaboration_target` null while a required field is empty, set when complete and thin, null once asked, still offering a different thin entry, and ignoring user-role text; system instruction contents and renumbering. A stubbed `call_gemini_json` also drove `run_agent_turn` end to end for both new paths: the ask turn leaves the profile untouched with `state_update {}`, and the write turn reports `updated_fields ["project_description"]` with `state_update` limited to `projects`.
**Result**: Deterministic layer passes in full. **Live verification is blocked and remains pending** — all 4 attempts (3 planned plus the approved retry) returned `429 RESOURCE_EXHAUSTED` on `GenerateRequestsPerDayPerProjectPerModel-FreeTier` (limit 20/day, `gemini-3.6-flash`). A second attempt 75 seconds later, after the error's own `retryDelay: 57s`, failed identically, confirming the daily cap rather than a per-minute window. No model tool choice was observed either way, so nothing is claimed about it.
**Notes**: The 429s cost no quota, and the fallback path degraded correctly each time (`ask_targeted_follow_up` with the generic question, profile unmodified). `_verify_elaboration_live.py` is kept in the repo, ready to re-run unchanged once the daily quota resets: scenario A asserts the elaboration tool fires and then that only `projects[0].description` changes, scenario B asserts no false trigger on substantive descriptions.

## Entry 46 — Diagnosis: two CVs for Maya against different jobs came out identical
**Goal**: Root-cause why two CVs for the same profile against two different jobs were byte-identical. Diagnosis only, no fix in this entry.
**Plan proposed**: Rule the session-state hypothesis in or out first by reading `_select_job` / `_render_jobs`, then test whether the document is job-invariant by construction.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: Read the selection path, then generated Maya's CV markdown against all 9 real jobs and hashed each, plus per-project overlap counts for her top 3 cards. 0 Gemini calls.
**Result**: **Not stale state.** `_select_job` re-derives description, summary, markdown, sections, `cv_section_order`, widget keys and `pdf_bytes` from the clicked job on every click, after `_clear_cv_widget_keys()`; `_render_jobs` passes each card's own dict. Nothing from today's `cv_pdf_sections` dirty tracking or the section-order override is implicated. The real cause is that Maya's document has no job-dependent surface left under current conditions: all 9 jobs yield one identical markdown (sha 439c4a51e1fa). Three compounding reasons. (1) The Professional Summary is the only genuinely job-tailored text, and `_templated_summary(profile)` takes no job input at all; with the free-tier daily quota exhausted (Entry 45's 429s), every `generate_cv_summary` call now returns that job-independent template. (2) Skills is job-independent by approved design since Entry 44's predecessor change. (3) Projects/Experience ordering works but cannot express a difference for this profile: overlap counts do vary per job (`agentic_ai_engineer` 1/1, `agentic_ai_data_scientist` 3/1, `rag_agent_engineer` 1/1) yet Team Course Planner is already first in her profile, so the ranking never flips, and with a single experience entry scoring 0 everywhere, Projects always leads.
**Notes**: This also explains why differentiation "worked earlier today" — Gemini was still answering then, and the earlier reorder proof used Jordan Lee against contrived vision-vs-NLP jobs whose technologies are disjoint, a case that can actually flip. Two side observations: the running Streamlit terminal captured here is stale (last output 2026-09-01 11:38), and per Entry 43 `_log_cv_diag`'s stderr writes are swallowed under Streamlit on Windows, so the `cv_summary path=` transparency line is effectively invisible in the app — worth revisiting if that evidence matters.

## Entry 47 — Job-aware fallback summary plus a visible summary-path caption
**Goal**: Fix Entry 46's two root causes: the fallback summary ignored the job entirely, and which path produced the summary was invisible in the app.
**Plan proposed**: (1) `_templated_summary(profile, job_description="", job_title="")` keeps every claim a profile fact and lets the job choose only what to highlight — the skills that job actually mentions (`_skills_mentioned_in_job`, capped at 3) and the highest-overlap project via `_most_relevant_project` (strict `>` so ties keep the earliest). With no job or no overlap it collapses to the old profile-only wording. `_short_job_title` drops the seniority tail after a `|`. (2) `generate_cv_summary` gains optional `job_title` and a `diagnostics` out-dict filled with `{path, reason, detail}`; `_select_job` stores it in `cv_summary_path` and the CV editor renders one caption. `_skills_mentioned_in_job` itself is unchanged, so Projects/Experience reordering still uses the same narrow job-specific list.
**Files changed**: `app/cv_generator.py`, `streamlit_app.py`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: Regenerated Maya's and Ethan's CVs against their top 3 cards and hashed each document; stubbed `call_gemini_json` for all four summary paths; template edge cases; AppTest with Gemini wrapped off. 0 Gemini calls.
**Result**: Pass, with one honest limit. Maya went from 1 distinct document across 9 jobs to 2 distinct across her 3 cards; Ethan likewise 2 of 3. Example: `agentic_ai_engineer` gives "Strengths relevant to the AI Engineer role include Python, evidenced in Team Course Planner", while `agentic_ai_data_scientist` gives "…the AI Engineer / Data Scientist (Agentic AI) role include Python, SQL, Git…". The pair that still matches is `rag_agent_engineer` and `agentic_ai_engineer`: they are two distinct postings that share the identical title "AI Engineer", and only "Python" of Maya's 14 skills appears in either description, so there is no profile-true difference to state. `healthcare_ml_engineer` and `recommendation_systems_ml_engineer` share the title "Machine Learning Engineer" the same way. Diagnostics recorded correctly for all four paths (`gemini`; `api` with `GeminiAPIError: 429 RESOURCE_EXHAUSTED`; `empty`; `ungrounded` with `['Kubernetes', 'led development']`), and the AppTest caption read "Professional Summary: deterministic template (no Gemini key configured)." with a 2442-byte PDF and no exception.
**Notes**: Two wording bugs were caught during verification and fixed: "relevant to the this role" when no title was supplied, and the full job-board title "AI Engineer / Data Scientist (Agentic AI) | Manager/Senior Manager" reading badly mid-sentence. Getting all three of Maya's cards to differ would need job-side text that is not a profile fact, which is exactly what the grounding rules forbid — that resolution is what the Gemini path buys, since it reads the whole description.

## Entry 49 — 0-call proof that confirmation is a Python gate
**Goal**: Show `finalize_and_confirm_profile` refuses an incomplete profile even when the model is stubbed to "obey" a confirmation injection. Live adversarial call held until after the daily quota reset (~5:00 PM AEST).
**Plan proposed**: Direct gate tests plus a stubbed `run_agent_turn` that returns `tool_name=finalize_and_confirm_profile`. 0 Gemini calls.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: `finalize_and_confirm_profile` on empty, name-only, and target_roles-only profiles; complete-profile negative control; stubbed injection through `run_agent_turn`; stubbed `retrieve_matching_jobs` on an unconfirmed complete profile.
**Result**: Pass. Empty and name-only return `target_roles required`. Target-roles-only returns `profile incomplete` with `missing_fields=['degree', 'institution', 'year', 'technical_skills', 'projects', 'experience']`. Complete profile still confirms. When the stubbed model called `finalize_and_confirm_profile` after the stacked injection, observation was `{'status': 'error', 'reason': 'target_roles required'}`, `is_confirmed` stayed `False`, and no profile fields changed. Unconfirmed `retrieve_matching_jobs` returned `profile must be confirmed first`.
**Notes**: The stubbed `assistant_message` was still `"Your profile is confirmed."` — that string is the model's phrasing, not Python state. The gate held. Live 1+3 calls remain pending until quota reset.

## Entry 48 — Chat UI for MAX_TURNS lock and Gemini fallback
**Goal**: Review what the chat actually shows when the step limit fires, and when Gemini is unavailable mid-conversation. UI only — no `agent_engine.py` logic change.
**Plan proposed**: Seed both states in AppTest (0 Gemini calls). If MAX_TURNS only hid the text box with no explanation, surface `_step_limit_reached`'s message. If fallback leaked errors, keep the chat as a normal follow-up.
**Files changed**: `streamlit_app.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: AppTest. Turn 15 with an unconfirmed profile (real `run_agent_turn`, which short-circuits before Gemini). Fallback via `get_gemini_client = lambda: None` and a raised `GeminiAPIError`. 0 Gemini calls.
**Result**: Neither path was silently dead. MAX_TURNS already wrote the last chat bubble ("The profile could not be completed in 15 turns. Still missing: technical_skills, projects, experience, target_roles.") and hid `st.chat_input` (`requires_input=False`). The lock banner was the weak part: generic "The agent is no longer accepting input for this profile." with no mention of turns or how to continue (the jobs-loaded banner already points at Start New Profile). Fallback already showed only the canned follow-up in chat ("What technical skills would you like listed on your profile?"); 429 / `FALLBACK:` text stayed in the sidebar Agent Trace. Applied the approved banner-only polish: when the last tool is `step_limit_reached`, the info line is now "We've reached the maximum number of turns for this session. The last message above lists what's still missing. Use Start New Profile in the sidebar to begin again." Re-verified: chat input still 0, last bubble unchanged, new banner present; fallback still a normal question with chat input open and no error text in the bubble.
**Notes**: One engine-side quirk left untouched by design: the 15th user message is not written into the profile because the limit check runs at the start of the turn, so a user who just typed skills can still see `technical_skills` in the missing list. That is MAX_TURNS behavior, not a display bug.

## Entry 50 — Override chat text when a gated tool returns error
**Goal**: Stop a false "Your profile is confirmed." (or similar) from appearing in chat when `finalize_and_confirm_profile` or `retrieve_matching_jobs` actually refused.
**Plan proposed**: Do not sniff success language in `assistant_message`. If `tool_name` is one of those two and `observation.status == "error"`, the chat bubble is built from `observation.reason` (and `missing_fields` when present). Success observations and every other tool keep the model text. Trace still stores the raw `assistant_message`.
**Files changed**: `streamlit_app.py`, `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: Helper unit checks; Entry 49 stub (`run_agent_turn` with a model that calls finalize and says "Your profile is confirmed."); AppTest chat bubble. 0 Gemini calls.
**Result**: Pass. Stub still has raw `assistant_message="Your profile is confirmed."` and `is_confirmed=False`. Chat displays `I couldn't confirm the profile yet (target_roles required).` Incomplete confirm lists missing fields. Unconfirmed retrieve displays `I couldn't retrieve matching jobs yet (profile must be confirmed first).` Successful confirm and `ask_targeted_follow_up` are unchanged.
**Notes**: Detection is status-based, not wording-based, so the model cannot keep a lying bubble by avoiding the word "confirmed".

## Entry 51 — Deterministic pytest suite
**Goal**: Commit the already-designed 0-Gemini checks as a repeatable `pytest tests/` suite. No new product behavior. No live Gemini.
**Plan proposed**: `tests/test_agent_engine.py`, `test_cv_generator.py`, `test_vector_store.py`, plus `test_streamlit_gated_chat.py` (Entry 50). `tests/conftest.py` autouse-patches `get_gemini_client` to `None` and `call_gemini_json` to raise if a test forgets to stub. Retrieval uses local MiniLM with Gemini wrapped off. Grounding tests `_ungrounded_summary_terms` directly; `generate_cv_summary` is stubbed.
**Files changed**: `tests/` (new), `pytest.ini`, `requirements.txt` (`pytest`), `README.md`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/` — 34 passed in ~45s. 0 Gemini calls.
**Result**: Pass. Agent gates, Entry 7 multi-field write, MAX_TURNS (LLM mock never invoked), Entry 49 injection stub, thin-content 7/8 boundary, CV confirmed-only, skill mining, layout threshold, grounding edge cases, local top-k retrieval, gated-chat override.
**Notes**: First retrieval test pays MiniLM load time. `search_top_jobs` would otherwise call Gemini three times for gap analysis; the autouse client stub keeps those as the fallback string.

## Entry 52 — Live ethics/robustness calls after quota reset
**Goal**: Run the 4 approved live calls from Entry 49 (1 adversarial confirmation + 3 gap analyses). Stop only on 429.
**Plan proposed**: Incomplete profile + stacked injection via `run_agent_turn`. Then `generate_gap_analysis` for the Entry 18 CV-tester profile vs `nlp_multimodal_ml_engineer`, `graduate_ai_engineer`, `healthcare_ml_engineer`.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 4 live Gemini calls. Key SET. None were 429.
**Result**:
- Adversarial: `is_confirmed` stayed `False`. The model never chose a tool — call 1 returned `503 UNAVAILABLE` (high demand), so `_fallback_turn` used `ask_targeted_follow_up` with observation `{"question": "Could you share your degree, the institution, and your graduation or study year?"}`. Not a gate-fire; the 0-call stub in Entry 49 remains the evidence that Python refuses if the model *does* call finalize.
- `nlp_multimodal_ml_engineer`: `503 UNAVAILABLE` → fallback `"Gap analysis unavailable for this match."` No prose to quality-check.
- `graduate_ai_engineer`: same 503 fallback.
- `healthcare_ml_engineer`: schema key `gap_analysis`. Text: "While the student brings valuable experience building and training computer vision models with PyTorch, YOLO, and video streams, they lack the required two to three years of software engineering experience and explicit familiarity with SQL, Linux, CI/CD, and model deployment/monitoring in production environments." Names real job requirements (2–3 years, SQL, Linux, CI/CD, deployment/monitoring). Student-side PyTorch/YOLO/video are in the CV-tester profile, not invented as job requirements.
**Notes**: Continued after 503 because the stop rule was 429 only. No retries. 4-call budget spent. Temporary `_verify_ethics_live.py` removed after the run.

## Entry 53 — Live elaboration follow-ups (3 turns, 503 retry once)
**Goal**: Confirm `ask_elaboration_question` fires on a thin project, the answer writes `project_description` without corrupting siblings, and substantive profiles get the optional activities prompt instead.
**Plan proposed**: Scenario A two turns on a constructed complete profile with projects[0] thin. Scenario B one turn with all descriptions substantive. Retry a 503 once; stop on 429.
**Files changed**: `docs/prompt_record.md` (this entry)
**Verification performed**: 5 live Gemini calls (3 planned + one 503 retry on A1 and A2). No 429.
**Result**: Pass.
- A1 (after 503 retry): `tool_name=ask_elaboration_question`, parameters `{item_type: project, item_title: Course planner}`, observation the exact template question, `state_update {}`, profile unchanged.
- A2 (after 503 retry): `update_profile_field` with `project_description {index: 0, description: "Built the timetable clash checker in Flask, wrote the SQL queries behind it and the unit tests, and presented the demo to our tutor."}`. Observation success. Three projects remain; siblings byte-identical; title/technologies intact; experience untouched; `detect_thin_content` empty.
- B1 (no retry): `tool_name=none`, assistant_message is `ACTIVITIES_PROMPT`. Not `ask_elaboration_question`.
**Notes**: Temporary `_verify_elaboration_live.py` removed after the run.

## Entry 54 — Formal 5-profile user-to-PDF suite (stopped on 429)
**Goal**: Brief requirement — complete user-to-PDF for all five profiles, including incomplete ones that need follow-ups. Skip-path 01/02/05 (0 conversation calls) then live `generate_cv_summary` vs prior #1 job. Live `run_agent_turn` for 03/04 from `initial_introduction` until `is_confirmed=True` (≥2 follow-up turns), then retrieve + CV.
**Plan proposed**: Cap 20 live calls (user-approved). Retrieval always wraps `get_gemini_client → None`. 503 retry once per turn/summary; stop on 429. 03/04 replies from JSONL only; commit roles only after the indecisive intro.
**Files changed**: `docs/prompt_record.md` (this entry). Temporary `_verify_e2e_five_profiles.py` and `_e2e_five_results.json` removed after the run.
**Verification performed**: 10 live `gemini-3.6-flash` calls, then 429 `RESOURCE_EXHAUSTED`. No gap_analysis calls.
**Result**: Partial. Skip-path CVs written for Maya, Liam, Ethan against the same #1 jobs as the predict-before-retrieve run (`agentic_ai_engineer` 0.594, `graduate_ai_engineer` 0.647, `agentic_ai_engineer` 0.600). All three summaries used the job-aware template (Maya: retry succeeded but ungrounded `AI-driven`; Liam/Ethan: 503 twice). No Target Roles / Growth Areas on any PDF. Sofia: intro correctly did not write `target_roles`; turn 2 wrote all required fields + JSONL data roles (`missing=[]`) but did not confirm; turn 3 429. Noah not started.
**Notes**: Call split: Maya 2, Liam 2, Ethan 2, Sofia 4, Noah 0. Seven 503s. Liam's template names the job title `Graduate AI Engineer`, which the profile-blob grounding helper flags — not invented profile facts. Resume after quota reset should skip 01/02/05 and re-run 03/04 from the intro (in-memory state was not persisted).

## Entry 55 — Job title is not an ungrounded summary claim
**Goal**: Stop the grounding checker treating the selected job title as a fabricated claim, so a summary that names its target role stays on the Gemini path, while invented skills/experience still fall back.
**Plan proposed**: Pass `job_title` into `_ungrounded_summary_terms` / `_extract_summary_claims`. Tokens that match the job title (including the short form before `|`) are not claims. Risky phrases and other proper-noun/tech claims still check the profile blob only. Hardcoded pytest; 0 Gemini calls.
**Files changed**: `app/cv_generator.py`, `tests/test_cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/` — 35 passed. New test: tailored "Graduate AI Engineer" summary is clean and `generate_cv_summary` records `path=gemini`; the same sentence plus Kubernetes / led development still flags those two. Autouse Gemini kill-switch in `tests/conftest.py` was not lifted.
**Result**: Pass. 0 live Gemini calls.
**Notes**: A skill that also appears in the job title would be allowed as a role name, not as profile evidence. Typical titles here (`Graduate AI Engineer`, `AI Engineer`) do not name tools.

## Entry 56 — Cross-user isolation (Maya vs Liam, 0 Gemini)
**Goal**: Prove two real profiles never mix in inspect/finalize/retrieve/CV, including interleaved PDF writes and two concurrent Streamlit sessions.
**Plan proposed**: Load `student_profile_01` / `02` as separate `StudentProfile` objects. Run inspect + finalize independently. Skip-path-confirm Liam (JSONL has empty experience). `search_top_jobs` with Gemini already stubbed. Interleave Maya markdown → Liam markdown → Maya PDF → Liam PDF. AST inventory of module-level caches. Two `AppTest` sessions with `pending_demo_action` set to 01 vs 02.
**Files changed**: `tests/test_cross_user_isolation.py` (created), `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/test_cross_user_isolation.py` then `pytest tests/` — 40 passed. 0 Gemini calls.
**Result**: Pass. Maya outputs contained only Maya identity (name, Team Course Planner, retail role); Liam only Liam (name, retrieval/classifier projects). Mutating Maya's match dict / session profile did not change Liam. Shared caches are the job index / MiniLM / Gemini client / demo catalog, none profile-keyed.
**Notes**: Liam `finalize_and_confirm_profile` correctly refuses (`experience` missing); skip-path still sets `is_confirmed=True` without copying Maya's experience. Risks that did not fire: `_demo_catalog` `lru_cache` holds mutable JSONL dicts (skip-path does not write back); `_search_top_jobs_without_gemini` monkeypatches `vector_store.get_gemini_client` at module level (restored in `finally`; a thread race could affect gap_analysis availability, not profile bytes).

## Entry 57 — Live 03/04 user-to-PDF after quota reset
**Goal**: Finish the 5-profile suite. Restart Sofia and Noah from `initial_introduction`, live `run_agent_turn` until confirmed (≥2 follow-ups), local retrieve, CV+PDF with `generate_cv_summary` under the job-title grounding fix.
**Plan proposed**: Cap 20 live calls for both. Intro only on turn 1 (no guessed `target_roles`). Later replies from JSONL. 503 retry once per call; stop on 429. Retrieval wraps Gemini off.
**Files changed**: `docs/prompt_record.md` (this entry). Temporary `_verify_e2e_03_04.py` and `_e2e_03_04_results.json` removed after the run.
**Verification performed**: 12 live `gemini-3.6-flash` calls. No 429. Two 503s (Sofia turn 3, Noah turn 1), each retried once successfully.
**Result**: Pass. Sofia: 4 turns, intro did not write roles, confirmed on turn 4, top job `graduate_ai_engineer` 0.613, summary `path=gemini`, PDF 2123 B, checker clean. Noah: 4 turns, intro did not write roles, confirmed on turn 4, top job `agentic_ai_engineer` 0.575, summary `path=gemini`, PDF 2527 B, checker clean. Follow-up turns: 3 each.
**Notes**: Call split Sofia 6, Noah 6. Sofia's Gemini summary names the intern titles as if they were held roles (`experience as a Data Analyst Intern…`); those strings are in `target_roles` so the checker did not reject them. Noah's summary stays on projects/skills. Neither PDF has Target Roles or Growth Areas. Skip-path 01/02/05 were not re-run.

## Entry 58 — Target-role employment framing is ungrounded
**Goal**: Stop summaries from treating `target_roles` as held jobs (`experience as a Data Analyst Intern`) while still allowing aspiration wording (`seeking` / `aiming for`).
**Plan proposed**: Prompt rule in `_SUMMARY_SYSTEM_INSTRUCTION`. Checker flags `experience as` / `worked as` / `experience in the role of` immediately before a title that appears only in `target_roles`, not in projects/experience. Hardcoded pytest, then one live Sofia `generate_cv_summary` vs `graduate_ai_engineer`.
**Files changed**: `app/cv_generator.py`, `tests/test_cv_generator.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/` — 41 passed. 0 Gemini for the tests. Live: first call 503, one retry → `path=gemini`.
**Result**: Pass. Hardcoded: "She has experience as a Data Analyst Intern" flags; "She is seeking a Data Analyst Intern role" is clean; held `Peer Learning Volunteer` with "experience as" is allowed. Live Sofia summary: "Aiming for a Data Analyst Intern, Business Intelligence Intern, or Junior Data Specialist role…" No employment framing. Ungrounded list empty.
**Notes**: Standing 503-retry-once used; 2 generate_content attempts, 1 successful summary. Temporary `_verify_sofia_summary.py` removed.

## Entry 59 — Maya two-job Evaluation 3.2 comparison
**Goal**: Side-by-side of Maya vs `agentic_ai_engineer` and `agentic_ai_data_scientist` for report section 3.2 (summary, section order, skills, projects, missing requirements).
**Plan proposed**: 4 live calls (2 `generate_cv_summary` + 2 `generate_gap_analysis`). Markdown/order/skills local. 503 retry once; stop on 429.
**Files changed**: `docs/prompt_record.md` (this entry). Temporary `_verify_maya_two_jobs.py` and `_maya_two_job_compare.json` removed after the run.
**Verification performed**: 4 live `gemini-3.6-flash` calls, all succeeded. No 503, no 429. Both summaries `path=gemini`, ungrounded empty, aspiration framing (`Seeking`).
**Result**: Summaries and gap_analyses differ by job. Skills list, project order, experience order, and section order are identical. Job-overlapping skills differ (`Python` vs `Python, SQL, Git`) but that did not change rank: Team Course Planner still first.
**Notes**: Gap text is job-card only, not on the CV. First gap names 2+ years Python, cloud, Terraform/Kubernetes, Docker, Langflow. Second names Azure, Docker, vector databases, RAG evaluation, production agentic systems.

## Entry 60 — Final consolidation (pins, README, evidence pack)
**Goal**: Pin direct dependencies, rewrite README for the finished system, remove the stale PENDING block, and assemble the A1 report materials folder. 0 Gemini calls.
**Plan proposed**: `requirements.txt` keeps only packages the app/tests import, pinned from `pip freeze`. README drops phase language. Prompt log: Entries 0–59 present (48/49 were written out of numeric sequence; both exist). PENDING removed. Submission tree copied without `.env` / `venv` / `__pycache__`. Three sample PDFs generated locally from recorded Maya summaries plus Liam's job-aware template.
**Files changed**: `requirements.txt`, `README.md`, `.gitignore`, `docs/prompt_record.md` (PENDING removed; this entry), `A1_Report_Materials_STUDENTID/` (created)
**Verification performed**: `pytest tests/` after the pack copy — 41 passed. Source files in the project root were not rewritten except the docs/requirements/README listed above. Pack `01_System/source_code/` hashes match the working app.
**Result**: Pass. Architecture diagram is a placeholder — drop `architecture_diagram.png` into `01_System/` when available. Rename `STUDENTID` in the folder name before submit.

## Entry 61 — Always attach the next student question after a silent update
**Goal**: After `update_profile_field` (or any silent turn), Python must ask the next question so the student is never left with a thank-you and a spinning “Thinking…” and no idea what to do. Prefer `ask_elaboration_question` when `next_elaboration` is set. Do not auto-confirm or auto-generate a CV.
**Plan proposed**: Deterministic `_next_actionable_prompt` / `_apply_follow_on_prompt` after the LLM tool. Priority: empty required field → thin project/experience (`ask_elaboration_question`) → optional activities prompt → confirm-to-retrieve. Skip gated tools and any `assistant_message` that already contains `?`. Trace `observation.follow_on` names the attached ask. Sidebar label shows `update → ask_elaboration_question` when that happens. 0 Gemini calls.
**Files changed**: `app/agent_engine.py`, `streamlit_app.py`, `tests/test_agent_engine.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/`
**Result**: Pass. 45 passed. Confirmation and retrieve stay Python gates. Job cards still appear only after confirm (`_maybe_load_jobs`); the CV still starts when the student picks a card.

## Entry 62 — Keep asking every thin entry; do not treat a stray “?” as done
**Goal**: After required fields are complete, Python must keep driving the next step (elaboration for each remaining thin item, then activities, then confirm) without waiting for “anything else?”. Do not let `finalize_and_confirm_profile` skip leftover thin entries unless the user explicitly confirms or skips.
**Plan proposed**: Diagnose first: (1) `_assistant_already_asks` treated any `?` in `assistant_message` as “already asked,” so a thank-you like “Anything else?” skipped both `ask_elaboration_question` and the optional activities prompt. (2) There is no multi-item loop in one turn (one tool per turn); continuation depends on the next turn’s follow-on, which was skipped by that `?` check or by a premature `finalize`. Fix: skip follow-on only when the message already contains the computed next question; rewrite a non-explicit finalize into `ask_elaboration_question` while `next_elaboration` is set. 0 Gemini calls.
**Files changed**: `app/agent_engine.py`, `tests/test_agent_engine.py`, `A1_Report_Materials_14722262/A1_Report_14722262.md` (one sentence in §2.2), `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/`
**Result**: Pass. 48 passed. A thank-you that ends in “Anything else?” now still attaches the next thin-item question or the activities prompt. A premature finalize is rewritten to `ask_elaboration_question` unless the user explicitly confirms or skips.

## Entry 63 — Completeness-flipping turn: same-turn follow-on, not a new root cause
**Goal**: Confirm whether the turn that writes the last required field appends the next student prompt in that same `assistant_message`, or only on a later turn that never happens.
**Plan proposed**: Read `_dispatch_tool` then `_apply_follow_on_prompt` / `_next_actionable_prompt`. Add a regression test: profile missing only `institution` and `year`; stub `update_profile_field` with a thank-you and no question; assert `ACTIVITIES_PROMPT` is in the same reply. 0 Gemini calls. No new product logic if the test already passes under Entries 61–62.
**Files changed**: `tests/test_agent_engine.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/test_agent_engine.py::test_completeness_flipping_update_appends_next_prompt_same_turn` then `pytest tests/`
**Result**: Pass. 49 passed. This is the same class as Entries 61–62, not a separate defer-to-next-turn bug. After the write, Python already inspects the updated profile on that turn and appends activities (or elaboration / confirm). A live “thanks only” reply after institution+year means the running app is pre-61/62, or required fields were not actually complete (so follow-on would have been a missing-field question — restart Streamlit and check the sidebar).

## Entry 64 — “confirm” must dispatch finalize_and_confirm_profile, not re-ask
**Goal**: Stop the confirmation loop. If the student replies `confirm` (or `yes it is` after the confirm prompt) and required fields are complete, Python must call `finalize_and_confirm_profile` even when the model returns `tool_name=none`.
**Plan proposed**: Diagnose: `awaiting_confirmation` is only `observation.follow_on.status`, not a state machine. Entries 61–63 append `CONFIRM_PROMPT` but never execute finalize. `_EXPLICIT_CONTINUE_RE` only allows finalize when thin items remain; it does not invoke it. When the model chooses `none`, follow-on appends the same prompt again. Fix: before dispatch, if `_user_confirms_profile` and required-complete and the tool is not an update/retrieve/finalize, force `finalize_and_confirm_profile`. Treat short affirmations as confirm only when the last assistant line was the confirm prompt. 0 Gemini calls.
**Files changed**: `app/agent_engine.py`, `tests/test_agent_engine.py`, `docs/prompt_record.md` (this entry)
**Verification performed**: `pytest tests/`
**Result**: Pass. 51 passed. Literal `confirm` and `yes it is` after the confirm prompt now dispatch `finalize_and_confirm_profile` when the model returns `none`. The confirm gate is still Python; injection still cannot set `is_confirmed` on an incomplete profile.












