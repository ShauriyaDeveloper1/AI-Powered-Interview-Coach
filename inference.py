"""
Submission inference entrypoint.

Validator requirements covered:
- Uses OpenAI client for all LLM calls.
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment.
- Runs on 3 tasks (easy, medium, hard) and emits reproducible score report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Load env vars from a local .env file when available.
if load_dotenv is not None:
    load_dotenv()

from openai import OpenAI

from rl_interview_coach import Action, FeedbackStrategy, InterviewCoachEnv, TaskBank, TaskType

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK") or "interview-coach"

MAX_ATTEMPTS = 3
REPORT_PATH = Path("reports/inference_scores.json")


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _format_reward_list(rewards: List[float]) -> str:
    return ",".join(f"{value:.2f}" for value in rewards)


def _sanitize_field(value: str) -> str:
    # Keep each log line single-line and parser-safe.
    return " ".join(str(value).split())


def _log_start(task_name: str) -> None:
    print(
        f"[START] task={_sanitize_field(task_name)} env={_sanitize_field(BENCHMARK)} model={_sanitize_field(MODEL_NAME or 'unknown')}",
        flush=True,
    )


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = _sanitize_field(error) if error else "null"
    print(
        f"[STEP] step={step} action={_sanitize_field(action)} reward={reward:.2f} done={_bool_str(done)} error={err}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={_bool_str(success)} steps={steps} score={score:.3f} rewards={_format_reward_list(rewards)}",
        flush=True,
    )


def _require_config() -> None:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def _choose_strategy(attempt: int) -> FeedbackStrategy:
    if attempt == 1:
        return FeedbackStrategy.HINT
    if attempt == 2:
        return FeedbackStrategy.MODERATE
    return FeedbackStrategy.STRICT


def _pick_tasks():
    return [
        TaskBank.get_tasks_by_difficulty(TaskType.EASY)[0],
        TaskBank.get_tasks_by_difficulty(TaskType.MEDIUM)[0],
        TaskBank.get_tasks_by_difficulty(TaskType.HARD)[0],
    ]


def _build_prompt(question: str, attempt: int, previous_feedback: List[str]) -> str:
    latest = "\n".join(previous_feedback[-2:]) if previous_feedback else "None"
    return (
        "Answer the interview question with concrete and concise details. "
        "Return only the answer text.\n\n"
        f"Question: {question}\n"
        f"Attempt: {attempt}\n"
        f"Previous feedback: {latest}\n"
        "Include specific actions and measurable outcomes when possible."
    )


def run_inference() -> Dict:
    _require_config()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = InterviewCoachEnv(seed=42, max_attempts=MAX_ATTEMPTS, target_grade=0.80)

    task_scores = []

    for task in _pick_tasks():
        env.reset(task)
        _log_start(task.task_id)

        best_grade = 0.0
        final_grade = 0.0
        total_reward = 0.0
        feedback_history: List[str] = []
        success = False
        attempts_used = 0
        step_rewards: List[float] = []

        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts_used = attempt
                strategy = _choose_strategy(attempt)
                prompt = _build_prompt(task.question, attempt, feedback_history)

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You produce interview answers only."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=220,
                )
                answer = ""
                if completion.choices and completion.choices[0].message.content:
                    answer = completion.choices[0].message.content.strip()

                action = Action(strategy=strategy, confidence=0.95, response_text=answer)
                step_result = env.step(action)

                step_reward = step_result.reward.total
                step_rewards.append(step_reward)
                _log_step(
                    step=attempt,
                    action=f"strategy({strategy.value})",
                    reward=step_reward,
                    done=step_result.done,
                    error=None,
                )

                final_grade = step_result.observation.current_grade
                best_grade = max(best_grade, final_grade)
                total_reward += step_reward

                feedback = step_result.info.get("feedback")
                if isinstance(feedback, str) and feedback:
                    feedback_history.append(feedback)

                if step_result.done:
                    success = step_result.reward.success
                    break
        except Exception as exc:
            attempts_used = max(1, attempts_used or 1)
            step_rewards.append(0.0)
            _log_step(
                step=attempts_used,
                action="exception",
                reward=0.0,
                done=True,
                error=str(exc),
            )
            success = False
        finally:
            task_score = final_grade if final_grade > 0 else best_grade
            _log_end(success=success, steps=attempts_used, score=task_score, rewards=step_rewards)

        task_scores.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty.value,
                "question": task.question,
                "attempts": attempts_used,
                "best_grade": round(best_grade, 4),
                "final_grade": round(final_grade, 4),
                "total_reward": round(total_reward, 4),
                "success": bool(success),
            }
        )

    aggregate_score = sum(item["final_grade"] for item in task_scores) / len(task_scores)
    success_rate = sum(1 for item in task_scores if item["success"]) / len(task_scores)

    report = {
        "api_base_url": API_BASE_URL,
        "model_name": MODEL_NAME,
        "seed": 42,
        "max_attempts": MAX_ATTEMPTS,
        "aggregate_score": round(aggregate_score, 4),
        "success_rate": round(success_rate, 4),
        "tasks": task_scores,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    run_inference()


if __name__ == "__main__":
    main()
