---
title: AI Powered Interview Coach
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
---

# AI-Powered Interview Coach

A full-stack Flask application combining interactive interview practice with reinforcement learning feedback policy optimization.

**Two components:**
1. **Web App** — Interview practice with text/audio/video modes and instant AI feedback.
2. **RL Environment** — OpenEnv-compliant `InterviewCoachEnv` for learning optimized coaching strategies.

## Environment Description and Motivation

Interview coaching is inherently iterative: *the same answer quality can improve or stagnate depending on feedback style*. This project models interview coaching as a sequential decision process where an RL agent learns which feedback strategy (strict, moderate, hint) maximizes answer improvement.

**Motivation:**
- Static, one-size-fits-all feedback is ineffective for personalized coaching.
- Different answer quality levels require different teaching strategies.
- RL enables data-driven policy learning: per-session, which coaching action best improves the next attempt?

**How it works:**
- **Episode:** One interview question session.
- **Observation:** Answer quality metrics (keyword coverage, sentiment, structure quality, grading metrics).
- **Action:** Agent selects feedback strategy (strict/moderate/hint).
- **Reward:** Grade improvement + efficiency bonuses/penalties.
- **Episode ends:** When target grade is reached or max attempts exhausted.

## Action Space

Defined in `rl_interview_coach/environment/models.py::Action`.

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `strategy` | enum | `"strict"`, `"moderate"`, `"hint"` | Feedback style selected by agent |
| `confidence` | float | `[0.0, 1.0]` | Agent confidence in this action |
| `response_text` | string (optional) | any text | Candidate answer (for `step(action)` compatibility) |

**Strategy semantics:**
- **Strict:** Detailed issue-focused feedback; points out all problems.
- **Moderate:** Balanced feedback; both strengths and improvement areas.
- **Hint:** Lightweight guidance for self-discovery; minimal directiveness.

## Observation Space

Defined in `rl_interview_coach/environment/models.py::Observation` and `openenv.yaml`.

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Task identifier (e.g., `"easy_001"`) |
| `difficulty` | enum | `"easy"` \| `"medium"` \| `"hard"` |
| `question` | string | Active interview prompt |
| `user_answer` | string | Most recent answer text |
| `answer_length` | int | Words in answer |
| `keywords_found` | int | Required keywords detected |
| `keyword_recall` | float | Keyword coverage ratio `[0.0, 1.0]` |
| `sentiment_score` | float | Sentiment polarity `[-1.0, 1.0]` |
| `structure_score` | float | Answer structure quality `[0.0, 1.0]` |
| `current_grade` | float | Overall answer grade `[0.0, 1.0]` |
| `attempt_number` | int | Current attempt index |
| `max_attempts` | int | Per-task attempt limit |
| `previous_feedback` | list[str] | Strategies used (history) |
| `improvement_history` | list[float] | Grade per attempt |

## Reward Definition

Computed in `rl_interview_coach/environment/env.py::_calculate_reward()`.

**Formula:** `total_reward = improvement_reward + efficiency_reward + max_attempts_penalty + reached_bonus`

- **Improvement reward:** `+10` for large grade gains; `-5` for no improvement or regression.
- **Efficiency reward:** `−0.5 × attempt_number` (penalty per attempt; encourages quick success).
- **Max attempts penalty/bonus:** `−5` if max attempts hit without target; `+2` if target reached at final attempt.
- **Reached bonus:** `+5 − (attempt_number × 0.5)` if target grade achieved.

## Task Descriptions with Expected Difficulty

Defined in `rl_interview_coach/environment/tasks.py::TaskBank`. Nine total tasks across three tiers.

| Tier | Count | Example Questions | Target Grade | Expected Difficulty |
|---|---:|---|---|---|
| **Easy** | 3 | "Tell me about yourself"; "Greatest strength/weakness?" | 0.75–0.80 | Short responses (25–30+ words), basic structure, low coaching overhead |
| **Medium** | 3 | "Technical challenge solved?"; "Team collaboration?"; "Why us?" | 0.75–0.80 | Longer (60–80+ words), STAR format, examples required, moderate coaching |
| **Hard** | 3 | "Conflict handling?"; "Biggest failure?"; "5-year vision?" | 0.78–0.82 | Long (80–100+ words), reflective/growth narrative, complex evaluation, high coaching skill needed |

## Highlights
- User authentication (signup/login) with local JSON storage
- Practice modes:
	- Audio response analysis (speech-to-text + feedback)
	- Text response analysis
	- Video interview practice with posture guidance
- Detailed evaluation across:
	- Tone/sentiment
	- Confidence
	- Professional vocabulary and filler words
	- Posture status and recommendations
- Mock interview video generator:
	- AI-generated interview transcript
	- Voice narration (interviewer/candidate style)
	- AI-generated realistic images
	- Final assembled MP4 video
- Progress tracking dashboards and downloadable PDF reports

## Tech Stack
- Backend: Flask (REST API)
- Frontend: HTML, CSS, JavaScript, Chart.js
- NLP: NLTK
- Speech: SpeechRecognition, sounddevice, wavio, pydub, gTTS
- Vision/Posture: OpenCV
- Video: MoviePy
- Data/Charts: pandas, matplotlib
- RL: Pydantic, PyYAML (OpenEnv spec)
- APIs: OpenAI-compatible endpoint (AIML API), Freepik image API

## Project Structure
```text
AI-Powered-Interview-Coach/
|- app.py                           # Flask web app
|- baseline.py                      # OpenAI baseline evaluation
|- inference.py                     # Model inference benchmark
|- openenv.yaml                     # OpenEnv specification
|- openenv.cmd                      # Windows validation shim
|- requirements.txt                 # Core dependencies
|- rl_interview_coach/
|  |- __init__.py                   # Package exports
|  |- agent/
|  |  |- ql_agent.py                # Q-Learning agent
|  |- environment/
|  |  |- env.py                     # Main RL environment
|  |  |- models.py                  # Pydantic models (Observation, Action, Reward)
|  |  |- tasks.py                   # Task bank definitions
|  |- graders/
|  |  |- answer_grader.py           # Answer evaluation logic
|- scripts/
|  |- openenv_cli.py                # OpenEnv validator CLI
|- static/                          # Frontend assets
|- templates/                       # HTML templates
|- reports/                         # User reports and run logs
|- models/                          # Saved agent checkpoints
|- users.json                       # User database
```

## Prerequisites

- **Python 3.10+**
- **FFmpeg** on PATH (recommended for audio/video features)

## Setup Instructions

### 1) Create virtual environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root:

```env
# For web app and baseline/inference scripts
API_KEY=your_openai_or_aiml_api_key
FREEPIK_API_KEY=your_freepik_api_key

# For AIML API (if using non-default endpoint)
API_BASE_URL=https://api.aimlapi.com/v1
# or
OPENAI_BASE_URL=https://api.aimlapi.com/v1

# For inference.py submission flow
MODEL_NAME=your_model_name
API_BASE_URL=https://your-proxy-base-url/v1
API_KEY=your-proxy-api-key

# Optional: Production security
FLASK_SECRET_KEY=your_secret_key_here
```

**Notes:**
- `.env` is auto-loaded by app.py and baseline.py (via python-dotenv).
- Add `.env` to `.gitignore` before committing.
- If `API_KEY` is unset, the app falls back to local transcript generation.
- If `FREEPIK_API_KEY` is unset, placeholder images are used instead.

## Usage Instructions

### Validate OpenEnv specification
```powershell
openenv validate openenv.yaml
```

On Windows without PATH shim:
```powershell
.\openenv.cmd validate openenv.yaml
```

### Run the web application
```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### Run baseline evaluation (requires `API_KEY` and `API_BASE_URL` if using AIML)
```powershell
python baseline.py
```

Output: `reports/baseline_openai_report.json` — deterministic score report on one task per difficulty tier.

### Run inference benchmark (requires `API_BASE_URL`, `API_KEY`, `MODEL_NAME`)
```powershell
python inference.py
```

Output: `reports/inference_scores.json` — model-specific benchmark metrics.

## Baseline Score

**Source:** Checkpoint trained on 50 training episodes (Q-Learning agent).

- Repository artifact: `models/agent_checkpoint.json`
- Episodes trained: `50`
- Total cumulative reward: `−47.2358`
- **Average reward per episode:** `−0.9447`

**Sample graded run:**
- Source: `reports/rlcheck03_reports.json`
- Task: "Describe a challenging technical problem you solved recently." (Medium tier)
- Grade achieved: `0.9278` (92.78% quality)
- Feedback strategy used: `hint`

*Note:* Negative average reward reflects the training challenge: early episodes explore ineffective strategies before convergence. Positive reward and high grades are achievable with tuned agents over more training episodes.

## Security

- Never commit real API keys.
- Keep `.env` local and gitignored.
- For production, use environment variable injection or secrets management.

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named moviepy.editor` | Use the pinned `moviepy==1.0.3` from requirements. |
| Browser speech recognition not working | Use a Chromium-based browser and ensure microphone permission is granted. |
| Video generation fails | Install FFmpeg and ensure it is available on PATH. |
| API errors (403/quota) | Verify key validity, quota, and billing on your provider account. |
| `API_KEY` not found (baseline/inference) | Create/update `.env` file in project root with your key. The app auto-loads it. |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request

## License

MIT License — see LICENSE file for details, or add before public distribution.
Made with ❤️ by team CodeSync

CodeSync:
- [Sarthak Maheshwari](https://github.com/Sarthak1Developer)
- [Shauriya Garg](https://github.com/ShauriyaDeveloper1)
