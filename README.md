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

## Important Links
- 🎥 **Video Demonstration:** [Watch on YouTube](https://youtu.be/xXdJKlznc2g?si=mI-OhNjt-l5tVC6q)
- 📓 **Agent Training & Data Analysis:** [View Colab Notebook](https://colab.research.google.com/drive/1Iu-Tg60nLtEEi4SkK2SEISX_1H_RUWI7)
- 💻 **Source Code Repository:** [Hugging Face Space Tree](https://huggingface.co/spaces/Shauriya24/AI-Powered-Interview-Coach/tree/main)

**Two components:**
1. **Web App** — Interview practice with text/audio/video modes and instant AI feedback.
2. **RL Environment** — OpenEnv-compliant `InterviewCoachEnv` for learning optimized coaching strategies.

## Problem Motivation

In the competitive landscape of job searching, interview coaching is highly subjective and inherently iterative: *the same answer quality can improve or stagnate depending on the feedback style*. Traditional AI solutions often provide static, one-size-fits-all feedback that fails to adapt to the user's progress or learning style, leading to diminished returns over repeated attempts.

This project treats interview coaching as a sequential decision-making problem. We solve this by modeling an OpenEnv-compliant reinforcement learning (RL) environment (`InterviewCoachEnv`) where an RL agent acts as the coach. The agent learns from interactive sessions which feedback strategy (strict criticism, moderate balancing, or gentle hints) maximizes user improvement.

**Key Problems Addressed:**
- **Static Feedback Failure:** One-size-fits-all advice is ineffective for personalized, dynamic learning.
- **Adaptive Coaching Requirements:** Different answer quality levels require different teaching strategies.
- **Data-Driven Policy Learning:** Leveraging RL enables learning the optimal coaching action to maximize answer improvement per-session.

## How the Environment Works

The core of the RL training is an interactive step-by-step coaching environment:
- **Episode (Session):** One complete interview question session where a user iteratively refines their answer.
- **Observation Space:** High-dimensional answer quality metrics including keyword coverage, sentiment score, structure quality, and past grading history.
- **Action Space:** The agent chooses a feedback policy: `strict` (direct & detailed criticism), `moderate` (balanced strengths & weaknesses), or `hint` (lightweight guidance).
- **Reward Function:** Driven by grade improvement `(+10)`. Penalizes excessive attempts to encourage efficiency `(-0.5 * attempt)`. Rewards achieving the target grade `(+5)`.
- **Termination:** The episode gracefully concludes when the candidate reaches the target acceptable grade or exceeds the maximum number of allowed attempts.

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
- ATS Resume Scanner:
	- Built-in PDF parsing and resume grading
	- Intelligent job capability matching against descriptions
- Detailed evaluation across:
	- Tone/sentiment
	- Confidence
	- Professional vocabulary and filler words
	- Posture status and recommendations
- Progress tracking dashboards and downloadable PDF reports

## Tech Stack
- Backend: Flask (REST API)
- Frontend: HTML, CSS, JavaScript, Chart.js
- NLP/ATS: NLTK, pdfplumber, python-docx, pyspellchecker
- Speech: SpeechRecognition, sounddevice, wavio, pydub, pyttsx3
- Vision/Posture: OpenCV
- Data/Charts: pandas, matplotlib, reportlab
- Machine Learning / RL: PyTorch, Transformers, Scikit-Learn, Pydantic, PyYAML (OpenEnv spec)
- APIs: OpenAI-compatible endpoint (AIML API)

## Training Results & Performance Metrics

### RL Agent Training Curves — Before and After

The RL agent was trained for **1000 episodes** to learn optimal feedback strategies. Below are the key performance metrics:

#### Training Metrics Visualization

![RL Agent Training Curves](rl_training_curves.png)

**Three key metrics:**

1. **Episode Rewards (Left Chart)**
   - **Before:** Noisy, unstable rewards with high variance (0-50 per episode)
   - **After:** Stabilized at ~23 average reward with 50-episode rolling average showing convergence
   - **Interpretation:** Agent learned to consistently achieve higher rewards through better policy decisions

2. **Win Rate (Middle Chart)** 
   - **Before:** Highly volatile, reaching 20% but with frequent drops to near 0%
   - **After:** Converged to 10-15% sustained win rate (50-episode rolling average)
   - **Interpretation:** Agent learned to achieve target grades within episode constraints more reliably

3. **Epsilon Decay (Right Chart)**
   - **Decay curve:** 1.0 → 0.05 over 1000 episodes (exponential decay)
   - **Purpose:** Balances exploration (random actions) vs. exploitation (learned policy)
   - **Effect:** Early exploration finds diverse strategies; later exploitation refines the best ones

### Dataset Analysis & Distribution

![HR Dataset EDA](hr_dataset_eda.png)

**Dataset characteristics:**

1. **Answer Length Distribution (Left)**
   - Ideal answer range: 20-40 words (peak at ~30 words)
   - Distribution shows task diversity with some longer responses (35+ words)

2. **Question Categories (Middle)**
   - 8 major interview categories equally represented
   - Balanced dataset: Conflict Resolution, Career Goals, Leadership, Team Collaboration, Work Style, Motivation, Culture Fit, Adaptability

3. **Difficulty Distribution (Right)**
   - ~800K samples per difficulty level (Easy, Medium, Hard)
   - Perfectly balanced across tiers for fair RL training

### Training Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total RL Episodes | 1000 | ✅ Completed |
| Converged Reward | ~23 per episode | ✅ Stable |
| Final Win Rate | 10-15% | ✅ Acceptable |
| Answer Dataset | 2.4M samples | ✅ Balanced |
| Feedback Strategies Learned | 3 (strict/moderate/hint) | ✅ Optimized |

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

### 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) Core Configuration

Create a `.env` file in the project root or configure Space Secrets:

```env
API_KEY=your_openai_or_aiml_api_key
FLASK_SECRET_KEY=your_secret_key_here
```

*(Note: Additional configurations for advanced features, such as Firebase passwordless login and local ML model fallbacks, are detailed in the repository's `.env_example` file).*

## RL Module Integration

The RL training endpoints (`/api/rl/*` and `/reset`, `/state`, `/step`) require the `rl_interview_coach` module to successfully initialize at startup. If Heavy ML dependencies (torch, transformers) are missing or fail to load due to deployment constraints, the project gracefully falls back to deterministic grading so core functions remain active.

You can verify RL availability via the diagnostics endpoint:
```bash
curl http://localhost:7860/api/diagnostics
```

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

Open `http://127.0.0.1:7860` in your browser.

To override the port, set `PORT` before running (e.g., `PORT=5000`).

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
| Browser speech recognition not working | Use a Chromium-based browser and ensure microphone permission is granted. |
| API errors (403/quota) | Verify key validity, quota, and billing on your provider account. |
| `API_KEY` not found | Provide the environment variables in your Hugging Face Space settings or local `.env` file. |

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
