"""Deterministic privacy-aware source-row selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

MINIMUM_DURATION_S = 3.0
MAXIMUM_DURATION_S = 12.0

_URL_RE = re.compile(r"(?i)(?:https?://|www\.)")
_EMAIL_RE = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){5,}(?!\w)")
_PRIVACY_TOKEN_RE = re.compile(
    r"(?i)\b(?:aadhar|aadhaar|social\s+security(?:\s+number)?|ssn|"
    r"passport(?:\s+number)?|driver'?s?\s+licen[cs]e|phone\s+number|"
    r"email(?:\s+address)?|home\s+address|postal\s+address|"
    r"bank\s+account(?:\s+number)?|account\s+number|credit\s+card|"
    r"debit\s+card|one[\s-]?time\s+password|\botp\b|password|"
    r"date\s+of\s+birth|\bdob\b|medical\s+record(?:\s+number)?|"
    r"\bmrn\b|patient\s+id|social\s+media(?:\s+(?:handle|username))?|"
    r"username|user\s+name)\b"
)
_DIGIT_RE = re.compile(r"\d", re.UNICODE)
_PRIVACY_TOKENS = (
    "aadhar",
    "aadhaar",
    "social security",
    "ssn",
    "passport",
    "driver's licence",
    "driver's license",
    "driving licence",
    "driving license",
    "phone number",
    "email address",
    "home address",
    "postal address",
    "bank account",
    "account number",
    "credit card",
    "debit card",
    "one time password",
    "otp",
    "password",
    "date of birth",
    "dob",
    "medical record number",
    "mrn",
    "patient id",
    "social media handle",
    "social media username",
    "username",
    "user name",
)


@dataclass(frozen=True)
class PrivacyDecision:
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectedFleursRow:
    row: dict[str, Any]
    duration_s: float
    privacy: PrivacyDecision


@dataclass(frozen=True)
class SelectedMucsSegment:
    recording_id: str
    start_sample: int
    end_sample: int
    transcript: str
    member_path: str
    row: dict[str, Any]
    duration_s: float
    privacy: PrivacyDecision


def privacy_reasons(transcript: object) -> tuple[str, ...]:
    """Return stable rejection reasons for a source transcript."""

    if not isinstance(transcript, str) or not transcript.strip():
        return ("empty_transcript",)
    reasons: list[str] = []
    if _DIGIT_RE.search(transcript):
        reasons.append("digit")
    if _URL_RE.search(transcript):
        reasons.append("url")
    if _EMAIL_RE.search(transcript):
        reasons.append("email")
    if _PHONE_RE.search(transcript):
        reasons.append("phone")
    normalized = " ".join(transcript.casefold().split())
    if _PRIVACY_TOKEN_RE.search(normalized) or any(
        token in normalized for token in _PRIVACY_TOKENS
    ):
        reasons.append("privacy_token")
    return tuple(reasons)


def _eligible_duration(value: object) -> float | None:
    if value is None:
        return None
    duration = float(value)
    if not MINIMUM_DURATION_S <= duration <= MAXIMUM_DURATION_S:
        return None
    return duration


def _transcript(row: Mapping[str, Any]) -> str:
    for key in ("transcription", "transcript", "text", "raw_transcription"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def select_fleurs_rows(
    rows: Iterable[Mapping[str, Any]],
    duration_probe: Callable[[Mapping[str, Any]], float | None],
    *,
    limit: int = 4,
) -> list[SelectedFleursRow]:
    """Select eligible validation rows by ascending numeric FLEURS ID."""

    if limit <= 0:
        raise ValueError("FLEURS selection limit must be positive")

    def numeric_id(row: Mapping[str, Any]) -> int:
        try:
            return int(row["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid FLEURS numeric id: {row!r}") from error

    ordered = sorted(rows, key=numeric_id)
    selected: list[SelectedFleursRow] = []
    for row in ordered:
        transcript = _transcript(row)
        reasons = privacy_reasons(transcript)
        if reasons:
            continue
        try:
            probed_duration = duration_probe(row)
        except Exception:
            probed_duration = None
        duration = _eligible_duration(probed_duration)
        if duration is None:
            continue
        selected.append(
            SelectedFleursRow(
                row=dict(row),
                duration_s=duration,
                privacy=PrivacyDecision(True, ()),
            )
        )
        if len(selected) == limit:
            return selected
    raise ValueError(
        f"only {len(selected)} eligible FLEURS rows found; required {limit}"
    )


def select_mucs_segments(
    segments: Iterable[Mapping[str, Any]],
    duration_probe: Callable[[Mapping[str, Any]], float | None],
    *,
    limit_recordings: int = 3,
) -> list[SelectedMucsSegment]:
    """Select one eligible segment from each of the first distinct recordings."""

    if limit_recordings <= 0:
        raise ValueError("MUCS recording limit must be positive")

    def sort_key(row: Mapping[str, Any]) -> tuple[str, int]:
        try:
            return (str(row["recording_id"]), int(row["start_sample"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid MUCS segment identity: {row!r}") from error

    selected: list[SelectedMucsSegment] = []
    admitted_recordings: set[str] = set()
    for row in sorted(segments, key=sort_key):
        recording_id = str(row["recording_id"])
        if recording_id in admitted_recordings:
            continue
        transcript = _transcript(row)
        reasons = privacy_reasons(transcript)
        if reasons:
            continue
        try:
            probed_duration = duration_probe(row)
        except Exception:
            probed_duration = None
        duration = _eligible_duration(probed_duration)
        if duration is None:
            continue
        start_sample = int(row["start_sample"])
        end_sample = int(row["end_sample"])
        member_path = str(
            row.get("member_path")
            or row.get("audio_path")
            or row.get("wav_path")
            or ""
        )
        selected.append(
            SelectedMucsSegment(
                recording_id=recording_id,
                start_sample=start_sample,
                end_sample=end_sample,
                transcript=transcript,
                member_path=member_path,
                row=dict(row),
                duration_s=duration,
                privacy=PrivacyDecision(True, ()),
            )
        )
        admitted_recordings.add(recording_id)
        if len(selected) == limit_recordings:
            return selected
    raise ValueError(
        f"only {len(selected)} eligible distinct MUCS recordings found; "
        f"required {limit_recordings}"
    )
