
"""Optional ML-powered feedback/grading.

This module is intentionally defensive:
- It uses local assets from `interview_coach_models/` by default.
- It avoids import-time failures when heavy deps (torch/transformers) are missing.
- It can operate in a "feedback-only" mode when classifier artifacts are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from . import get_models_dir


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class MLAnswerGrader:
    """Local model-backed grader.

    Assets (relative to `model_dir`):
    - `t5_interview_coach/` (HuggingFace T5 files)
    - `emotion_classifier.pkl` (optional)
    - `interview_classifier.pkl`, `feature_scaler.pkl`, `label_encoder.pkl`, `model_meta.json` (optional)
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = (Path(model_dir) if model_dir else get_models_dir()).resolve()

        self._joblib = None
        self._numpy = None
        self._torch = None
        self._t5_tokenizer = None
        self._t5_model = None
        self._t5_device = None
        self._t5_attempted = False

        # Optional classic ML artifacts
        self.clf = None
        self.scaler = None
        self.le = None
        self.meta: dict[str, Any] = {}

        # Optional emotion classifier
        self.emo_clf = None

        self._load_optional_artifacts()

    def _load_optional_artifacts(self) -> None:
        # joblib is used for .pkl artifacts.
        try:
            import joblib  # type: ignore

            self._joblib = joblib
        except Exception:
            self._joblib = None

        # Emotion classifier (small, optional)
        emo_path = self.model_dir / "emotion_classifier.pkl"
        if self._joblib is not None and emo_path.exists():
            try:
                self.emo_clf = self._joblib.load(str(emo_path))
            except Exception:
                self.emo_clf = None

        # Classic classifier bundle (optional)
        clf_path = self.model_dir / "interview_classifier.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        le_path = self.model_dir / "label_encoder.pkl"
        meta_path = self.model_dir / "model_meta.json"
        if self._joblib is not None and clf_path.exists() and scaler_path.exists() and le_path.exists():
            try:
                import numpy as np  # type: ignore

                self._numpy = np
                self.clf = self._joblib.load(str(clf_path))
                self.scaler = self._joblib.load(str(scaler_path))
                self.le = self._joblib.load(str(le_path))
                if meta_path.exists():
                    self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._numpy = None
                self.clf = None
                self.scaler = None
                self.le = None
                self.meta = {}

    def _ensure_t5_loaded(self) -> None:
        if self._t5_model is not None and self._t5_tokenizer is not None:
            return

        if self._t5_attempted:
            return

        self._t5_attempted = True

        t5_dir = self.model_dir / "t5_interview_coach"
        if not t5_dir.exists():
            return

        try:
            import torch  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

            self._torch = torch
            self._t5_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Prefer AutoTokenizer so local tokenizer.json works without SentencePiece.
            self._t5_tokenizer = AutoTokenizer.from_pretrained(str(t5_dir), use_fast=True)
            self._t5_model = AutoModelForSeq2SeqLM.from_pretrained(str(t5_dir)).to(self._t5_device)
            self._t5_model.eval()
        except Exception:
            self._torch = None
            self._t5_device = None
            self._t5_tokenizer = None
            self._t5_model = None

    def _heuristic_grade(self, keyword_recall: float, structure_score: float, word_count: int) -> float:
        # Conservative heuristic when the classic classifier bundle is absent.
        length_score = _clamp01(min(1.0, word_count / 120.0))
        return _clamp01(0.40 * keyword_recall + 0.30 * structure_score + 0.30 * length_score)

    def _predict_grade(self, keyword_recall: float, structure_score: float, word_count: int) -> float:
        if not (self.clf and self.scaler and self.le and self._numpy is not None):
            return self._heuristic_grade(keyword_recall, structure_score, word_count)

        np = self._numpy
        total_score = (keyword_recall * 10 + structure_score * 5 + word_count / 20)
        conf_score = min(12.0, total_score * 0.8)
        struct_score = min(9.0, structure_score * 9)
        fluency_score = min(9.0, keyword_recall * 9)
        features = np.array(
            [
                [
                    conf_score,
                    struct_score,
                    fluency_score,
                    total_score,
                    conf_score / (total_score + 1e-9),
                    struct_score / (total_score + 1e-9),
                    fluency_score / (total_score + 1e-9),
                ]
            ]
        )
        feat_scaled = self.scaler.transform(features)
        probs = self.clf.predict_proba(feat_scaled)[0]
        verdict_grade_map = {
            "Premium Select": 1.0,
            "Select": 0.85,
            "Borderline Select": 0.65,
            "Borderline Reject": 0.40,
            "Reject": 0.15,
        }
        grade = 0.0
        for idx, cls in enumerate(self.le.classes_):
            grade += _safe_float(probs[idx]) * _safe_float(verdict_grade_map.get(cls, 0.5), 0.5)
        return _clamp01(grade)

    def _generate_t5_feedback(self, question: str, answer: str) -> Optional[str]:
        self._ensure_t5_loaded()
        if not (self._t5_model and self._t5_tokenizer and self._torch and self._t5_device):
            return None

        prompt = f"evaluate answer: Q: {question} A: {answer[:300]}"
        try:
            enc = self._t5_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            enc = enc.to(self._t5_device)
            with self._torch.no_grad():
                out = self._t5_model.generate(**enc, max_length=128, num_beams=4)
            return self._t5_tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception:
            return None

    def _predict_emotion(self, answer: str) -> str:
        if not self.emo_clf:
            return "neutral"
        try:
            emo_id = int(self.emo_clf.predict([answer])[0])
        except Exception:
            return "neutral"

        emo_map = {0: "neutral", 1: "surprise", 2: "fear", 3: "sadness", 4: "joy", 5: "disgust", 6: "anger"}
        return emo_map.get(emo_id, "neutral")

    def grade(self, question: str, answer: str, keywords: list[str] | None = None) -> dict[str, Any]:
        keywords = keywords or []
        words = (answer or "").lower().split()
        word_count = len(words)

        found = 0
        if keywords:
            found = sum(1 for kw in keywords if any(str(kw).lower() in w for w in words))
        keyword_recall = found / max(len(keywords), 1)
        structure_score = min(1.0, word_count / 80)

        grade = self._predict_grade(keyword_recall, structure_score, word_count)
        feedback = self._generate_t5_feedback(question or "", answer or "")
        emotion = self._predict_emotion(answer or "")

        if not feedback:
            feedback = "Local ML feedback unavailable (missing torch/transformers or model files)."

        return {
            "grade": round(float(grade), 4),
            "keyword_recall": round(float(keyword_recall), 4),
            "keywords_found": int(found),
            "answer_length": int(word_count),
            "structure_score": round(float(structure_score), 4),
            "ml_feedback": str(feedback),
            "emotion": str(emotion),
        }


_CACHED_GRADER: Optional[MLAnswerGrader] = None
_CACHED_ERROR: Optional[str] = None


def get_ml_answer_grader() -> Optional[MLAnswerGrader]:
    """Return a cached MLAnswerGrader instance if it can be constructed."""

    global _CACHED_GRADER, _CACHED_ERROR
    if _CACHED_GRADER is not None:
        return _CACHED_GRADER
    if _CACHED_ERROR is not None:
        return None
    try:
        _CACHED_GRADER = MLAnswerGrader()
        return _CACHED_GRADER
    except Exception as exc:
        _CACHED_ERROR = str(exc)
        return None
