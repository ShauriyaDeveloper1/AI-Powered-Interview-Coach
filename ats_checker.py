from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class ExtractMeta:
    file_ext: str
    size_bytes: int
    pages: int | None = None
    docx_tables: int | None = None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\x0b\x0c]+", " ", text)
    return text.strip()


def _extract_text_pdf(path: str) -> tuple[str, ExtractMeta]:
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: pdfplumber. Install it to analyze PDFs.") from exc

    text_parts: list[str] = []
    pages = 0
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text_parts.append(page_text)

    text = _normalize_text("\n\n".join(text_parts))
    meta = ExtractMeta(file_ext=".pdf", size_bytes=os.path.getsize(path), pages=pages)
    return text, meta


def _extract_text_docx(path: str) -> tuple[str, ExtractMeta]:
    try:
        import docx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: python-docx. Install it to analyze DOCX files.") from exc

    doc = docx.Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if (p.text or "").strip()]
    text = _normalize_text("\n".join(paras))
    tables_count = len(getattr(doc, "tables", []) or [])
    meta = ExtractMeta(file_ext=".docx", size_bytes=os.path.getsize(path), docx_tables=tables_count)
    return text, meta


def _extract_text_txt(path: str) -> tuple[str, ExtractMeta]:
    with open(path, "rb") as f:
        raw = f.read()
    # Best-effort decode.
    for enc in ("utf-8", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = ""
    text = _normalize_text(text)
    meta = ExtractMeta(file_ext=".txt", size_bytes=os.path.getsize(path))
    return text, meta


def extract_resume_text(path: str, filename: str) -> tuple[str, ExtractMeta]:
    ext = os.path.splitext(filename or path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT.")

    if ext == ".pdf":
        return _extract_text_pdf(path)
    if ext == ".docx":
        return _extract_text_docx(path)
    return _extract_text_txt(path)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']{2,}", text or "")


def _count_metrics(text: str) -> int:
    # Counts common metrics-like patterns: numbers, percentages, currency, and shorthand.
    patterns = [
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",  # 1,000 or 12,345.67
        r"\b\d+(?:\.\d+)?%\b",  # 12%
        r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?\b",  # $1200
        r"\b\d+(?:\.\d+)?\s*(?:k|m|b)\b",  # 10k, 2.5m
        r"\b\d+(?:\.\d+)?\b",  # any number
    ]
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text, flags=re.IGNORECASE))
    # De-duplicate a bit: raw number pattern overlaps.
    return max(0, count - len(re.findall(patterns[-1], text, flags=re.IGNORECASE)) // 2)


def _lexical_diversity(words: list[str]) -> float:
    if not words:
        return 0.0
    low = [w.lower() for w in words]
    return _safe_div(len(set(low)), len(low))


def _repetition_score(text: str) -> tuple[float, int]:
    w = [x.lower() for x in _words(text)]
    w = [x for x in w if x not in _STOPWORDS]
    if len(w) < 80:
        return 1.0, 0

    diversity = _lexical_diversity(w)
    # Penalize very low diversity.
    score = _clamp01((diversity - 0.33) / 0.27)

    # Detect repeated lines (copy/paste blocks).
    lines = [ln.strip().lower() for ln in (text or "").splitlines() if ln.strip()]
    repeated_lines = sum(1 for _, c in Counter(lines).items() if c >= 3)

    issues = 0
    if score < 0.55:
        issues += 1
    if repeated_lines >= 1:
        issues += 1

    # If we found repeated lines, nudge score down.
    if repeated_lines:
        score = _clamp01(score - 0.15)

    return score, min(issues, 1)


def _spell_grammar_score(text: str) -> tuple[float, int]:
    words = [w for w in _words(text) if len(w) >= 4]
    if len(words) < 120:
        return 1.0, 0

    misspell_rate = 0.0
    try:
        from spellchecker import SpellChecker  # type: ignore

        spell = SpellChecker(distance=1)
        candidates = [w.lower() for w in words if w.isalpha()]
        sample = candidates[:2500]
        unknown = spell.unknown(sample)
        misspell_rate = _safe_div(len(unknown), max(1, len(set(sample))))
    except Exception:
        # If spellchecker isn't installed, don't block the feature.
        misspell_rate = 0.0

    # Lightweight grammar-ish heuristics.
    bad_punct = len(re.findall(r"[.?!]{2,}", text or ""))
    missing_space = len(re.findall(r"[a-z]\.[A-Z]", text or ""))
    grammar_penalty = min(0.12, 0.02 * bad_punct + 0.01 * missing_space)

    score = 1.0 - min(0.85, misspell_rate * 6.5) - grammar_penalty
    score = _clamp01(score)

    issues = 1 if score < 0.78 else 0
    return score, issues


def _parse_rate_score(text: str, meta: ExtractMeta) -> tuple[float, int]:
    # Parse rate is primarily about extractability. For TXT/DOCX, if we can read text,
    # treat it as fully parseable. For PDF, estimate extractability by text density.
    chars = len(text or "")

    if meta.file_ext in {".txt", ".docx"}:
        score = 1.0 if chars >= 50 else 0.0
        return score, (0 if score >= 1.0 else 1)

    pages = max(int(meta.pages or 1), 1)
    chars_per_page = chars / pages
    score = _clamp01(chars_per_page / 1800.0)
    issues = 1 if score < 0.65 else 0
    return score, issues


def _quantifying_impact_score(text: str) -> tuple[float, int]:
    metrics = _count_metrics(text or "")
    score = _clamp01(metrics / 10.0)
    issues = 1 if score < 0.5 else 0
    return score, issues


def _essential_sections_score(text: str) -> tuple[float, int]:
    t = (text or "").lower()
    essentials = [
        ("experience", ["experience", "work experience", "employment"]),
        ("education", ["education", "academics"]),
        ("skills", ["skills", "technical skills", "skills & tools"]),
        ("projects", ["projects", "project"]),
        ("summary", ["summary", "profile", "objective"]),
    ]

    found = 0
    for _name, keys in essentials:
        if any(k in t for k in keys):
            found += 1

    score = _clamp01(found / 5.0)

    # Require at least Experience + Education + Skills for a clean pass.
    required = ["experience", "education", "skills"]
    has_required = all(r in t for r in required)
    issues = 0 if has_required else 1
    return score, issues


_EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RX = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"
)


def _contact_info_score(text: str) -> tuple[float, int]:
    email = bool(_EMAIL_RX.search(text or ""))
    phone = bool(_PHONE_RX.search(text or ""))

    score = (1.0 if email else 0.0) * 0.6 + (1.0 if phone else 0.0) * 0.4
    issues = 0 if (email and phone) else 1
    return _clamp01(score), issues


def _file_format_size_score(filename: str, meta: ExtractMeta) -> tuple[float, int]:
    ext = meta.file_ext
    allowed = ext in ALLOWED_EXTENSIONS
    # ATS tools often struggle with huge files; keep a conservative default.
    size_ok = meta.size_bytes <= 2 * 1024 * 1024

    score = (1.0 if allowed else 0.0) * 0.6 + (1.0 if size_ok else 0.0) * 0.4
    issues = 0 if (allowed and size_ok) else 1
    return _clamp01(score), issues


def _design_score(text: str, meta: ExtractMeta) -> tuple[float, int]:
    # Heuristics: tables in DOCX are often ATS-unfriendly; PDFs with low parse rate also hint heavy design.
    score = 1.0

    if meta.file_ext == ".docx" and (meta.docx_tables or 0) > 0:
        score -= 0.55

    # Penalize for many lines with large gaps (pseudo-columns).
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    column_like = sum(1 for ln in lines[:120] if re.search(r"\S\s{6,}\S", ln))
    if column_like >= 6:
        score -= 0.35

    score = _clamp01(score)
    issues = 1 if score < 0.75 else 0
    return score, issues


def _email_score(text: str) -> tuple[float, int]:
    ok = bool(_EMAIL_RX.search(text or ""))
    return (1.0 if ok else 0.0), (0 if ok else 1)


def _hyperlink_header_score(text: str) -> tuple[float, int]:
    # Check the "header" portion: first few lines.
    header = "\n".join((text or "").splitlines()[:12]).lower()
    ok = any(
        k in header
        for k in (
            "http://",
            "https://",
            "linkedin.com",
            "github.com",
            "portfolio",
        )
    )
    return (1.0 if ok else 0.0), (0 if ok else 1)


def analyze_resume_file(path: str, filename: str) -> dict[str, Any]:
    text, meta = extract_resume_text(path, filename)

    guidance = {
        "ATS Parse Rate": {
            "what": "The resume text may not be fully readable by an ATS (common with scanned PDFs or heavy design).",
            "how": "Use a text-based PDF/DOCX, avoid scanned images, and ensure content is selectable/copyable.",
        },
        "Quantifying Impact": {
            "what": "The resume has limited measurable impact (numbers, percentages, costs, time saved).",
            "how": "Add metrics to bullets (e.g., +15%, -2 days, $10K saved, 3x throughput) where possible.",
        },
        "Repetition": {
            "what": "The resume repeats the same phrases/lines too often, which reduces clarity.",
            "how": "Rewrite repeated bullets, merge duplicates, and vary action verbs while keeping content concise.",
        },
        "Spelling & Grammar": {
            "what": "Spelling/grammar patterns may reduce ATS and recruiter confidence.",
            "how": "Run a spellcheck, keep tense consistent, and avoid punctuation/spacing errors.",
        },
        "Essential Sections": {
            "what": "Some expected resume sections may be missing or hard to detect.",
            "how": "Add clear headings like EXPERIENCE, EDUCATION, SKILLS, and optionally PROJECTS/SUMMARY.",
        },
        "Contact Information": {
            "what": "Key contact details (email/phone) are missing or not clearly detectable.",
            "how": "Add a professional email and phone number near the top of the resume.",
        },
        "File Format & Size": {
            "what": "The file format or size may not be ATS-friendly.",
            "how": "Upload a PDF/DOCX/TXT and keep the file under ~2MB when possible.",
        },
        "Design": {
            "what": "Layout features like tables/columns can confuse ATS parsers.",
            "how": "Avoid tables, text boxes, and multi-column layouts; use simple headings and bullets.",
        },
        "Email Address": {
            "what": "No email address was detected.",
            "how": "Add an email like name@example.com in the header.",
        },
        "Hyperlink in Header": {
            "what": "No portfolio/LinkedIn/GitHub hyperlink was detected in the top header.",
            "how": "Add a LinkedIn/GitHub/portfolio URL near your name at the top.",
        },
    }

    content_checks = [
        ("ATS Parse Rate", _parse_rate_score(text, meta)),
        ("Quantifying Impact", _quantifying_impact_score(text)),
        ("Repetition", _repetition_score(text)),
        ("Spelling & Grammar", _spell_grammar_score(text)),
    ]

    section_checks = [
        ("Essential Sections", _essential_sections_score(text)),
        ("Contact Information", _contact_info_score(text)),
    ]

    essentials_checks = [
        ("File Format & Size", _file_format_size_score(filename, meta)),
        ("Design", _design_score(text, meta)),
        ("Email Address", _email_score(text)),
        ("Hyperlink in Header", _hyperlink_header_score(text)),
    ]

    def build_group(group_id: str, label: str, checks: list[tuple[str, tuple[float, int]]]):
        items = []
        subscores = []
        issues = 0
        for check_label, (subscore, issue) in checks:
            subscores.append(float(subscore))
            issues += int(issue)
            guide = guidance.get(check_label, {})
            items.append(
                {
                    "label": check_label,
                    "status": "issue" if issue else "ok",
                    "issues": int(issue),
                    "score": round(float(subscore), 4),
                    "what": guide.get("what") if issue else "",
                    "how": guide.get("how") if issue else "",
                }
            )
        pct = int(round(100.0 * _safe_div(sum(subscores), len(subscores)))) if subscores else 0
        return {
            "id": group_id,
            "label": label,
            "percent": pct,
            "issues": int(issues),
            "items": items,
        }

    groups = [
        build_group("content", "CONTENT", content_checks),
        build_group("sections", "SECTIONS", section_checks),
        build_group("ats_essentials", "ATS ESSENTIALS", essentials_checks),
    ]

    # Weighted overall score.
    weights = {"content": 0.4, "sections": 0.3, "ats_essentials": 0.3}
    overall = 0.0
    total_w = 0.0
    total_issues = 0
    for g in groups:
        w = float(weights.get(g["id"], 0.0))
        overall += w * (float(g["percent"]) / 100.0)
        total_w += w
        total_issues += int(g.get("issues") or 0)

    overall = _safe_div(overall, total_w) if total_w else 0.0
    score = int(round(100.0 * _clamp01(overall)))

    word_count = len(_words(text))
    return {
        "ok": True,
        "score": score,
        "issues": int(total_issues),
        "groups": groups,
        "meta": {
            "filename": filename,
            "file_ext": meta.file_ext,
            "size_bytes": meta.size_bytes,
            "word_count": word_count,
            "pages": meta.pages,
            "docx_tables": meta.docx_tables,
        },
    }
