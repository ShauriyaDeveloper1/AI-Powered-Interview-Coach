"""
Submission inference entrypoint.

Validator requirements covered:
- Uses OpenAI client for all LLM calls.
- Reads API_BASE_URL, API_KEY, MODEL_NAME from environment.
- Runs on 3 tasks (easy, medium, hard) and emits reproducible score report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
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


def _log_bootstrap_failure(error: str) -> None:
    """Emit a minimal structured episode when a fatal error happens before task execution."""
    _log_start("bootstrap")
    _log_step(step=1, action="bootstrap-failure", reward=0.0, done=True, error=error)
    _log_end(success=False, steps=1, score=0.0, rewards=[0.0])


def _has_remote_config() -> bool:
    return bool(OpenAI and MODEL_NAME and os.getenv("API_BASE_URL") and os.getenv("API_KEY"))


def _normalize_api_key_env() -> None:
    """Support validator setups that provide HF_TOKEN instead of API_KEY."""
    if "API_KEY" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["API_KEY"] = os.environ["HF_TOKEN"]


def _require_proxy_env() -> None:
    """Require proxy variables in submission mode."""
    # Use exact validator variable names; no aliases/fallback providers.
    os.environ["API_BASE_URL"]
    os.environ["API_KEY"]


def _create_proxy_client():
    """Create OpenAI client bound only to injected validator proxy vars."""
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    return OpenAI(
        base_url=os.environ["API_BASE_URL"].strip(),
        api_key=os.environ["API_KEY"].strip(),
    )


def _preflight_proxy_call(client) -> None:
    """Make one mandatory proxy call so validator can observe LiteLLM key usage."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Reply with exactly: ok"},
            {"role": "user", "content": "ok"},
        ],
        max_tokens=4,
    )
    if not (completion.choices and completion.choices[0].message.content):
        raise RuntimeError("Proxy preflight returned empty content.")


def _get_answer(client, prompt: str) -> str:
    if client is None:
        raise RuntimeError("Proxy client is not available.")

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You produce interview answers only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=220,
    )
    if completion.choices and completion.choices[0].message.content:
        return completion.choices[0].message.content.strip()
    raise RuntimeError("LLM returned empty content in proxy mode.")


def _choose_strategy(attempt: int) -> str:
    if attempt == 1:
        return "hint"
    if attempt == 2:
        return "moderate"
    return "strict"


def _pick_tasks(task_bank, task_type):
    return [
        task_bank.get_tasks_by_difficulty(task_type.EASY)[0],
        task_bank.get_tasks_by_difficulty(task_type.MEDIUM)[0],
        task_bank.get_tasks_by_difficulty(task_type.HARD)[0],
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
    _normalize_api_key_env()
    _require_proxy_env()

    remote_mode = True
    key_source = "none"
    if os.getenv("API_KEY"):
        key_source = "API_KEY"

    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")

    print(
        f"[CONFIG] proxy_base_url_present={_bool_str(bool(api_base_url))} proxy_key_source={key_source} remote_mode={_bool_str(remote_mode)}",
        flush=True,
    )

    client = _create_proxy_client()
    _preflight_proxy_call(client)

    try:
        from rl_interview_coach import Action, InterviewCoachEnv, TaskBank, TaskType
    except Exception as exc:
        raise RuntimeError(f"Failed to import environment package: {exc}")

    env = InterviewCoachEnv(seed=42, max_attempts=MAX_ATTEMPTS, target_grade=0.80)

    task_scores = []

    for task in _pick_tasks(TaskBank, TaskType):
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

                answer = _get_answer(client, prompt)

                action = Action(strategy=strategy, confidence=0.95, response_text=answer)
                step_result = env.step(action)

                step_reward = step_result.reward.total
                step_rewards.append(step_reward)
                _log_step(
                    step=attempt,
                    action=f"strategy({strategy})",
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
        "api_base_url": api_base_url,
        "model_name": MODEL_NAME,
        "proxy_key_present": bool(api_key),
        "remote_mode": remote_mode,
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
    try:
        run_inference()
    except Exception as exc:
        _log_bootstrap_failure(_sanitize_field(str(exc)))


if __name__ == "__main__":
    main()
