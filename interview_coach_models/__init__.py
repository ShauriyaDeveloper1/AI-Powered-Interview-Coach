"""Local model assets bundled with the project.

This package stores offline/embedded model artifacts (e.g. T5 weights, classifiers).
Runtime code should treat these assets as optional and degrade gracefully when
heavy ML dependencies (torch/transformers) are not installed.
"""

from __future__ import annotations

from pathlib import Path
import os


def get_models_dir() -> Path:
    """Return the directory containing the bundled model artifacts.

    Override with `INTERVIEW_COACH_MODELS_DIR` to point at an external model cache.
    """

    override = (os.getenv("INTERVIEW_COACH_MODELS_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent
