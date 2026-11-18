"""Async structured-output VLM adapters for Ollama (offline) and Gemini Flash (BYOK)."""
import base64
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, field_validator

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class EventSearchResult(BaseModel):
    matched: bool = Field(description="True if the contact sheet clearly shows what the user asked for.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0.")
    best_timestamp: str = Field(default="", description="Exact HH:MM:SS timestamp burned onto the matching frame.")
    summary: str = Field(default="", description="Concise 1-sentence description of what happened in the frames.")
    tags: list[str] = Field(default_factory=list, description="2 to 4 short lowercase keyword tags.")

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("tags")
    @classmethod
    def _norm_tags(cls, v: list[str]) -> list[str]:
        return [str(t).strip().lower() for t in v if str(t).strip()][:4]


SYSTEM_PROMPT = """You are a home CCTV forensic assistant analyzing a 2x3 sequence of video frames.
Each tile has a burned-in timestamp (HH:MM:SS) in the top-left corner.
Evaluate the frames strictly against the user's search query.
Return ONLY valid JSON matching this schema:
{
  "matched": boolean,
  "confidence": float (0.0 to 1.0),
  "best_timestamp": "HH:MM:SS",
  "summary": "1-sentence description of the activity",
  "tags": ["tag1", "tag2"]
}"""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_result(raw_text: str) -> EventSearchResult:
    m = _FENCE_RE.match(raw_text)
    return EventSearchResult.model_validate_json(m.group(1) if m else raw_text.strip())


def _encode_image_b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _user_text(user_query: str) -> str:
    return f"User query: '{user_query}'. Analyze this CCTV contact sheet."


async def analyze_contact_sheet_ollama(
    image_path: str,
    user_query: str,
    model: str = "qwen2.5-vl:7b",
    ollama_url: str = OLLAMA_CHAT_URL,
    client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    """100% local offline VLM inference via Ollama."""
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_text(user_query), "images": [_encode_image_b64(image_path)]},
        ],
    }
    async with (client or httpx.AsyncClient(timeout=60.0)) as c:
        resp = await c.post(ollama_url, json=payload)
        resp.raise_for_status()
        return parse_result(resp.json()["message"]["content"])


async def analyze_contact_sheet_gemini(
    image_path: str,
    user_query: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    """Cloud BYOK inference via Gemini Flash. Only the contact-sheet JPEG is transmitted."""
    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "parts": [
                {"text": _user_text(user_query)},
                {"inlineData": {"mimeType": "image/jpeg", "data": _encode_image_b64(image_path)}},
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    # Key in header (not URL) so it never appears in logs or exception messages.
    headers = {"x-goog-api-key": api_key}
    async with (client or httpx.AsyncClient(timeout=30.0)) as c:
        resp = await c.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return parse_result(raw_text)
