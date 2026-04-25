"""
Deterministic graders for interview answers.
Scores answers from 0.0 to 1.0 based on:
- Keyword presence
- Answer length
- Structure (especially STAR format)
- Sentiment analysis
"""
import os
import re
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
# Epsilon for strict (0, 1) range
_EPSILON = 0.0005

def _clamp_score(score: float) -> float:
    """Clamp score to strictly within (0, 1) range."""
    return max(_EPSILON, min(1.0 - _EPSILON, score))
# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))


def _local_models_enabled() -> bool:
    value = (os.getenv("INTERVIEW_COACH_ENABLE_LOCAL_MODELS") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _grade_source() -> str:
    return (os.getenv("INTERVIEW_COACH_GRADE_SOURCE") or "deterministic").strip().lower()


def _try_enrich_with_local_ml(details: Dict, question: str, answer: str, keywords: List[str]) -> None:
    """Attach optional ML metadata into the `details` dict.

    This never raises and will silently no-op when dependencies are missing.
    """

    if not _local_models_enabled():
        return

    try:
        from interview_coach_models.ml_answer_grader import get_ml_answer_grader
    except Exception:
        return

    try:
        grader = get_ml_answer_grader()
        if grader is None:
            return
        result = grader.grade(question=question or "", answer=answer or "", keywords=keywords or [])
    except Exception:
        return

    try:
        details["ml_grade"] = float(result.get("grade"))
    except Exception:
        pass
    ml_feedback = result.get("ml_feedback")
    if isinstance(ml_feedback, str) and ml_feedback.strip():
        details["ml_feedback"] = ml_feedback.strip()
    emotion = result.get("emotion")
    if isinstance(emotion, str) and emotion.strip():
        details["emotion"] = emotion.strip()


class AnswerGrader:
    """Base grader with common utility methods."""
    
    @staticmethod
    def tokenize_answer(answer: str) -> List[str]:
        """Tokenize and clean answer."""
        try:
            tokens = word_tokenize(answer.lower())
            # Remove punctuation and stopwords
            tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
            return tokens
        except:
            return answer.lower().split()
    
    @staticmethod
    def get_sentiment_score(answer: str) -> float:
        """
        Get sentiment score using VADER.
        Range: -1 (very negative) to +1 (very positive)
        """
        scores = sia.polarity_scores(answer)
        return scores['compound']  # -1 to 1
    
    @staticmethod
    def get_length_score(answer: str, min_words: int = 20, max_words: int = 200) -> float:
        """
        Score based on answer length.
        - Too short (< min_words): score = 0.3
        - Optimal (min_words to max_words): score = 0.95 (not 1.0)
        - Too long (> max_words): score gradually decreases
        """
        words = answer.split()
        word_count = len(words)
        
        if word_count < min_words:
            return _clamp_score(0.3)
        elif word_count <= max_words:
            return _clamp_score(0.95)
        else:
            # Penalize for being too verbose
            excess = word_count - max_words
            return _clamp_score(max(0.6, 1.0 - (excess / max_words) * 0.3))


class GeneralAnswerGrader(AnswerGrader):
    """Grader for general interview questions."""
    
    def __init__(self):
        self.keywords = {
            "Tell me about yourself.": [
                "experience", "skills", "background", "education", "achievement", 
                "interested", "passionate", "expertise"
            ],
            "What is your greatest strength?": [
                "strength", "excel", "skilled", "talented", "leadership", 
                "communication", "problem-solving", "analytical"
            ],
            "What is your greatest weakness?": [
                "weakness", "challenge", "improve", "learning", "growing",
                "addressed", "overcome"  # Should mention growth
            ],
            "Why do you want to work for this company?": [
                "company", "mission", "culture", "values", "role",
                "opportunity", "growth", "contribute", "aligned"
            ],
        }
    
    def grade(self, question: str, answer: str) -> Tuple[float, Dict]:
        """
        Grade a general interview answer (0.0 to 1.0).
        Returns (score, details).
        """
        details = {}
        scores = []
        
        # 1. Keyword recall (30% weight)
        keywords = self.keywords.get(question, [])
        keyword_score, keywords_found = self._keyword_score(answer, keywords)
        details['keyword_score'] = keyword_score
        details['keywords_found'] = keywords_found
        details['total_keywords'] = len(keywords)
        scores.append(('keyword', keyword_score, 0.3))
        
        # 2. Length score (20% weight)
        length_score = self.get_length_score(answer, min_words=30, max_words=150)
        details['length_score'] = length_score
        scores.append(('length', length_score, 0.2))
        
        # 3. Sentiment score (15% weight) - should be positive
        sentiment = self.get_sentiment_score(answer)
        sentiment_score = (sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
        details['sentiment_score'] = sentiment
        details['sentiment_score_normalized'] = sentiment_score
        scores.append(('sentiment', sentiment_score, 0.15))
        
        # 4. Coherence score (20% weight) - check for multiple sentences
        coherence_score = self._coherence_score(answer)
        details['coherence_score'] = coherence_score
        scores.append(('coherence', coherence_score, 0.2))
        
        # 5. Structure bonus (15% weight) - check for examples/specifics
        structure_score = self._structure_score(answer)
        details['structure_score'] = structure_score
        scores.append(('structure', structure_score, 0.15))
        
        # Weighted average, clamped to strict (0, 1)
        final_score = sum(s * w for _, s, w in scores)
        final_score = _clamp_score(final_score)

        _try_enrich_with_local_ml(details, question=question, answer=answer, keywords=keywords)
        if _grade_source() in {"ml", "local_ml", "local"} and isinstance(details.get("ml_grade"), float):
            final_score = _clamp_score(details["ml_grade"])

        details['final_score'] = final_score
        
        return final_score, details
    
    @staticmethod
    def _keyword_score(answer: str, keywords: List[str]) -> Tuple[float, int]:
        """Score based on keyword presence."""
        if not keywords:
            return _clamp_score(1.0), 0
        
        answer_lower = answer.lower()
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        score = found / len(keywords)
        return _clamp_score(score), found
    
    @staticmethod
    def _coherence_score(answer: str) -> float:
        """Score based on having multiple sentences."""
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return _clamp_score(0.4)
        elif len(sentences) >= 4:
            return _clamp_score(0.95)
        else:
            return _clamp_score(0.6 + (len(sentences) - 2) * 0.2)
    
    @staticmethod
    def _structure_score(answer: str) -> float:
        """Check for specific examples or structured content."""
        indicators = [
            r'(for example|for instance|such as|specifically)',
            r'(i\s+(worked|developed|built|created|managed))',
            r'(result|outcome|achieved|delivered)',
            r'(percentage|\d+%|number|\d+\+)',
        ]
        
        found = sum(1 for pattern in indicators if re.search(pattern, answer.lower()))
        return _clamp_score(min(0.95, found / len(indicators) + 0.3))


class BehavioralAnswerGrader(AnswerGrader):
    """Grader for behavioral (STAR format) questions."""
    
    def __init__(self, question: str | None = None):
        self.question = question or ""
        self.star_keywords = {
            'situation': [
                'was', 'were', 'when', 'where', 'team', 'project', 'challenge'
            ],
            'task': [
                'responsible', 'assigned', 'needed', 'had to', 'required', 'task'
            ],
            'action': [
                'i ', 'i\'d', 'did', 'took', 'implemented', 'developed', 'created',
                'analyzed', 'proposed', 'suggested', 'led', 'managed'
            ],
            'result': [
                'result', 'outcome', 'improved', 'increased', 'decreased', 'achieved',
                'succeeded', 'completed', 'delivered', 'percent', '%', 'saved'
            ]
        }
    
    def grade(self, answer: str) -> Tuple[float, Dict]:
        """
        Grade a behavioral (STAR format) answer (0.0 to 1.0).
        Returns (score, details).
        """
        details = {}
        
        # Check STAR structure
        star_scores = self._check_star_structure(answer)
        details.update(star_scores)
        
        # Check for quantifiable results (hard metrics)
        has_metrics = self._has_metrics(answer)
        details['has_metrics'] = has_metrics
        
        # Length check
        length_score = self.get_length_score(answer, min_words=80, max_words=300)
        details['length_score'] = length_score
        
        # Overall weighting:
        # STAR structure: 50%
        # Metrics: 20%
        # Length: 15%
        # Sentiment/professionalism: 15%
        
        star_avg = sum(star_scores.values()) / len(star_scores)
        metric_score = 0.95 if has_metrics else _clamp_score(0.5)
        sentiment = _clamp_score((self.get_sentiment_score(answer) + 1) / 2)
        
        final_score = (
            star_avg * 0.5 +
            metric_score * 0.2 +
            length_score * 0.15 +
            sentiment * 0.15
        )
        final_score = _clamp_score(final_score)

        flat_keywords: List[str] = []
        try:
            for group in self.star_keywords.values():
                flat_keywords.extend([str(v) for v in group])
        except Exception:
            flat_keywords = []

        _try_enrich_with_local_ml(details, question=self.question, answer=answer, keywords=flat_keywords)
        if _grade_source() in {"ml", "local_ml", "local"} and isinstance(details.get("ml_grade"), float):
            final_score = _clamp_score(details["ml_grade"])

        details['final_score'] = final_score
        return final_score, details
    
    def _check_star_structure(self, answer: str) -> Dict[str, float]:
        """Check presence of STAR components."""
        answer_lower = answer.lower()
        
        scores = {}
        for component, keywords in self.star_keywords.items():
            found = sum(1 for kw in keywords if kw in answer_lower)
            scores[f'star_{component}_score'] = _clamp_score(min(0.95, found / max(len(keywords), 1)))
        
        return scores
    
    @staticmethod
    def _has_metrics(answer: str) -> bool:
        """Check for quantifiable results (numbers, percentages, etc.)."""
        metric_patterns = [
            r'\d+%',
            r'\$\d+',
            r'\d+\s*(million|thousand|k)',
            r'(\d+\.\d+|\d+)\s*(x|times|percent|%)',
            r'increased|decreased|improved|reduced.*\d+',
        ]
        return any(re.search(pattern, answer, re.IGNORECASE) for pattern in metric_patterns)


class TechnicalAnswerGrader(AnswerGrader):
    """Grader for technical interview questions."""
    
    def __init__(self):
        self.quality_indicators = [
            'algorithm', 'complexity', 'trade-off', 'optimization',
            'testing', 'edge case', 'scalability', 'performance',
            'design', 'architecture'
        ]
    
    def grade(self, question: str, answer: str) -> Tuple[float, Dict]:
        """Grade a technical answer (0.0 to 1.0)."""
        details = {}
        
        # Technical depth
        depth_score = self._check_technical_depth(answer)
        details['technical_depth'] = depth_score
        
        # Clarity
        clarity_score = self._check_clarity(answer)
        details['clarity'] = clarity_score
        
        # Length
        length_score = self.get_length_score(answer, min_words=50, max_words=300)
        details['length_score'] = length_score
        
        # Code samples or pseudo-code
        code_score = self._has_code_sample(answer)
        details['has_code'] = code_score
        
        final_score = (
            depth_score * 0.3 +
            clarity_score * 0.3 +
            length_score * 0.2 +
            code_score * 0.2
        )
        final_score = _clamp_score(final_score)

        _try_enrich_with_local_ml(details, question=question, answer=answer, keywords=[])
        if _grade_source() in {"ml", "local_ml", "local"} and isinstance(details.get("ml_grade"), float):
            final_score = _clamp_score(details["ml_grade"])

        details['final_score'] = final_score
        return final_score, details
    
    def _check_technical_depth(self, answer: str) -> float:
        """Check for technical depth indicators."""
        answer_lower = answer.lower()
        found = sum(1 for indicator in self.quality_indicators 
                   if indicator in answer_lower)
        return _clamp_score(min(0.95, found / len(self.quality_indicators)))
    
    @staticmethod
    def _check_clarity(answer: str) -> float:
        """Check for clarity (good sentence structure, NOT too long sentences)."""
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return _clamp_score(0.1)
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Ideal: 10-20 words per sentence
        if avg_length < 5:
            return _clamp_score(0.4)
        elif avg_length <= 20:
            return _clamp_score(0.95)
        else:
            # Too long sentences
            return _clamp_score(max(0.5, 1.0 - (avg_length - 20) / 30))
    
    @staticmethod
    def _has_code_sample(answer: str) -> float:
        """Check for code examples or pseudocode."""
        code_indicators = [
            r'```',
            r'\bfunction\b|\bdef\b|\bclass\b',
            r'=>|->|:=',
            r'if\s*\(|while\s*\(|for\s*\(',
        ]
        found = sum(1 for pattern in code_indicators if re.search(pattern, answer))
        return _clamp_score(min(0.95, found / len(code_indicators) * 0.8 + 0.2))


def get_grader(question: str, difficulty: str) -> Tuple[AnswerGrader, str]:
    """
    Factory function to get the appropriate grader based on question type.
    """
    behavioral_questions = [
        "describe a challenging situation",
        "challenging situation at work",
        "conflict",
        "failure",
        "success",
        "proud of",
    ]
    
    technical_questions = [
        "how would you",
        "design",
        "implement",
        "optimization",
        "architecture",
        "algorithm",
    ]
    
    question_lower = question.lower()
    
    # Determine question type
    if any(bq in question_lower for bq in behavioral_questions):
        return BehavioralAnswerGrader(question=question), "behavioral"
    elif any(tq in question_lower for tq in technical_questions):
        return TechnicalAnswerGrader(), "technical"
    else:
        return GeneralAnswerGrader(), "general"
