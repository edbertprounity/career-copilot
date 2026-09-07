# Student Career Copilot

UTS 43007 Advanced AI assessment project. A Streamlit app that helps a student build a confirmed career profile through a multi-turn Gemini agent, retrieve matching jobs with local embeddings, generate a fact-bound CV, edit it section by section, and export an A4 PDF. Sessions are isolated: one browser session never sees another user's profile.

The build is complete. There is no remaining "phase in progress."

## What the system does

1. **Profile agent** — Six Python tools collect and confirm name, education, skills, projects, experience, optional activities, and target roles. The model chooses one tool per turn; Python executes it.
2. **Job retrieval** — After confirmation, `retrieve_matching_jobs` ranks `data/jobs.jsonl` (9 postings) with MiniLM embeddings and cosine similarity. Top 3 matches are shown. Gap analysis is Gemini text on the job card only; it is not printed on the CV.
3. **Fact-bound CV** — Markdown and PDF are built only from confirmed profile facts. Skills are the full genuine set, grouped into categories. Layout is adaptive (two-column when richness ≥ 8). The professional summary is Gemini when available, otherwise a job-aware template. Grounding rejects invented skills, risky seniority phrases, job-title false positives, and employment framing of aspiration-only `target_roles`.
4. **Edit and export** — The user can edit sections and download an A4 PDF.
5. **Session isolation** — Profile, chat, jobs, and CV state live in `st.session_state`. Shared caches are the job index and embedding model only.

## Agent tools

| Tool | Role |
|---|---|
| `update_profile_field` | Write one or more profile fields, including index-addressed `project_description` / `experience_description` |
| `inspect_completeness` | Report missing required fields (Python) |
| `ask_targeted_follow_up` | Ask for missing fields (name/degree/institution/year are grouped) |
| `ask_elaboration_question` | Ask for detail on a thin project or experience description |
| `finalize_and_confirm_profile` | Confirm only if `target_roles` is set **and** `inspect_completeness` is complete |
| `retrieve_matching_jobs` | Rank jobs; refused until `is_confirmed` is true |

Confirmation is a Python gate. The LLM cannot confirm an incomplete profile. After a field write, Python attaches the next student question on the same turn (missing field, thin project/experience, optional activities, or “reply confirm”). An explicit `confirm` (or `yes` after that prompt) dispatches `finalize_and_confirm_profile` even if the model returned `none`. The agent hard-stops at `MAX_TURNS` (15).

## Demo cards

On an empty session the app offers five JSONL profiles plus a blank start:

- **01 Maya, 02 Liam, 05 Ethan** — `clear after confirmation`. Skip-path: load the JSONL row, set `is_confirmed=True`, retrieve jobs with Gemini wrapped off.
- **03 Sofia, 04 Noah** — `requires a target-role follow-up`. The real `initial_introduction` is sent into `run_agent_turn`. Target roles are not written while the student is still deciding.

## Constraints

- Completeness and confirmation are deterministic Python.
- The CV does not invent content and does not print Target Roles or Growth Areas.
- One tool per turn. Full trace: thought → tool → observation → state update.
- Free-tier Gemini (`gemini-3.6-flash`) is used for agent turns, summaries, and gap analysis. Retrieval ranking does not need Gemini.

## Folder structure

```
career-copilot/
├── app/
│   ├── llm_client.py       # Cached Gemini client + JSON generate_content
│   ├── schemas.py          # StudentProfile, ProjectRecord, ExperienceRecord, AgentTraceStep
│   ├── agent_engine.py     # Six tools, run_agent_turn, MAX_TURNS
│   ├── vector_store.py     # MiniLM + cosine retrieval + gap_analysis
│   └── cv_generator.py     # Fact-bound markdown, adaptive PDF, grounded summary
├── data/
│   ├── jobs.jsonl
│   ├── student_profiles.jsonl
│   └── student_introductions.jsonl
├── tests/                  # Deterministic pytest; Gemini is blocked
├── docs/prompt_record.md   # Sequential build log (Entries 0–64)
├── streamlit_app.py
├── requirements.txt        # Pinned direct dependencies
├── config.example
├── .gitignore
└── README.md
```

## Setup

1. Create and activate a virtual environment from the project root:

   ```bash
   python -m venv venv
   ```

   Windows (PowerShell):

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `config.example` to `.env` and put your Gemini API key in `GEMINI_API_KEY`. Never commit `.env`.

   ```bash
   copy config.example .env
   ```

## How to run

```bash
streamlit run streamlit_app.py
```

Use the project venv so `sentence-transformers` resolves (`.\venv\Scripts\python.exe -m streamlit run streamlit_app.py` on Windows). The app calls the agent, retrieval, and CV helpers in-process. Chat locks once jobs are loaded.

## Tests

```bash
pytest tests/
```

The suite never makes a live Gemini call (`tests/conftest.py` kill-switch).

Optional client smoke test (uses the key; one live call):

```bash
python -m app.llm_client
```
