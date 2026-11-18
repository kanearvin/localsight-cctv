"""Async structured-output VLM adapters: Ollama (offline), Gemini, and OpenAI-compatible BYOK providers."""
import asyncio
import base64
import os
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, field_validator

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
# Ollama library tag (no hyphen): https://ollama.com/library/qwen2.5vl
# 3B default: the 7B model only partially fits on GPU on 8 GB Macs and times out.
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"
# Generous per-request timeout: CPU-offloaded local VLMs can take minutes per contact sheet.
OLLAMA_TIMEOUT_SEC = float(os.environ.get("LOCALSIGHT_OLLAMA_TIMEOUT", "300"))

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# gemini-2.5-flash is deprecated (404 for new keys); Google recommends these for new projects.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite", "gemini-4-hydrogen")
GEMINI_TIMEOUT_SEC = 60.0

# ---- OpenAI-compatible BYOK providers (free tiers available) ----
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
# Free (":free") vision models change often; the UI loads the live list via /api/openrouter/models.
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"

CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
DEFAULT_CLOUDFLARE_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"
CLOUDFLARE_MODELS: tuple[str, ...] = (
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/google/gemma-3-12b-it",
)
OPENAI_COMPAT_TIMEOUT_SEC = 90.0
RATE_LIMIT_RETRIES = 2  # free tiers return 429 under burst load


class VLMProviderError(RuntimeError):
    """Upstream VLM returned an HTTP error; message carries the provider's own explanation."""


def _raise_for_status(resp: httpx.Response, provider: str) -> None:
    if resp.is_success:
        return
    detail = ""
    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        detail = (err.get("message") if isinstance(err, dict) else err) or ""
        # Cloudflare API style: {"errors": [{"code": 10000, "message": "Authentication error"}]}
        if not detail and isinstance(body, dict) and isinstance(body.get("errors"), list):
            detail = "; ".join(str(e.get("message", e)) for e in body["errors"] if e)
    except Exception:
        detail = resp.text[:200]
    raise VLMProviderError(f"{provider} HTTP {resp.status_code}: {detail or resp.reason_phrase}".strip())


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
    text = m.group(1) if m else raw_text.strip()
    try:
        return EventSearchResult.model_validate_json(text)
    except Exception:
        # Smaller/free models sometimes wrap the JSON in prose; fall back to the outermost object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return EventSearchResult.model_validate_json(text[start : end + 1])


def _encode_image_b64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _user_text(user_query: str) -> str:
    return f"User query: '{user_query}'. Analyze this CCTV contact sheet."


async def analyze_contact_sheet_ollama(
    image_path: str,
    user_query: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = OLLAMA_CHAT_URL,
    client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    """100% local offline VLM inference via Ollama."""
    payload = {
        "model": model,
        "format": "json",
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_text(user_query), "images": [_encode_image_b64(image_path)]},
        ],
    }
    async with (client or httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SEC)) as c:
        resp = await c.post(ollama_url, json=payload)
        _raise_for_status(resp, "Ollama")
        return parse_result(resp.json()["message"]["content"])


async def analyze_contact_sheet_gemini(
    image_path: str,
    user_query: str,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
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
    async with (client or httpx.AsyncClient(timeout=GEMINI_TIMEOUT_SEC)) as c:
        resp = await c.post(url, json=payload, headers=headers)
        _raise_for_status(resp, f"Gemini ({model})")
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return parse_result(raw_text)


def _message_text(content: object) -> str:
    """OpenAI-style message content may be a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _retry_after_sec(resp: httpx.Response, attempt: int) -> float:
    try:
        return min(30.0, max(0.5, float(resp.headers.get("retry-after", ""))))
    except ValueError:
        return min(30.0, 2.0 * (2**attempt))


async def analyze_contact_sheet_openai_compatible(
    image_path: str,
    user_query: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider_name: str,
    extra_headers: dict[str, str] | None = None,
    json_mode: bool = True,
    client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    """Chat-completions call with an inline base64 JPEG. Only the contact sheet is transmitted."""
    payload: dict = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _user_text(user_query)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_b64(image_path)}"}},
                ],
            },
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}
    url = f"{base_url.rstrip('/')}/chat/completions"

    async with (client or httpx.AsyncClient(timeout=OPENAI_COMPAT_TIMEOUT_SEC)) as c:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            resp = await c.post(url, json=payload, headers=headers)
            if resp.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
                await asyncio.sleep(_retry_after_sec(resp, attempt))
                continue
            break
        _raise_for_status(resp, f"{provider_name} ({model})")
        body = resp.json()
        # Some gateways report upstream failures with HTTP 200 and an "error" object.
        if isinstance(body, dict) and body.get("error") and not body.get("choices"):
            err = body["error"]
            raise VLMProviderError(f"{provider_name} ({model}): {err.get('message') if isinstance(err, dict) else err}")
        text = _message_text(body["choices"][0]["message"].get("content"))
        if not text.strip():
            raise VLMProviderError(f"{provider_name} ({model}) returned an empty response")
        return parse_result(text)


async def analyze_contact_sheet_openrouter(
    image_path: str, user_query: str, api_key: str, model: str = DEFAULT_OPENROUTER_MODEL,
    client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    return await analyze_contact_sheet_openai_compatible(
        image_path, user_query, base_url=OPENROUTER_BASE_URL, api_key=api_key, model=model,
        provider_name="OpenRouter",
        # Optional app attribution headers recognised by OpenRouter (no user data).
        extra_headers={"X-Title": "LocalSight CCTV"},
        client=client,
    )


async def analyze_contact_sheet_cloudflare(
    image_path: str, user_query: str, api_key: str, account_id: str,
    model: str = DEFAULT_CLOUDFLARE_MODEL, client: httpx.AsyncClient | None = None,
) -> EventSearchResult:
    return await analyze_contact_sheet_openai_compatible(
        image_path, user_query, base_url=CLOUDFLARE_BASE_URL.format(account_id=account_id),
        api_key=api_key, model=model, provider_name="Cloudflare Workers AI", client=client,
    )


def filter_free_vision_models(models: list[dict]) -> list[dict]:
    """From OpenRouter's /models list, keep free chat models that accept images and emit text."""
    out = []
    for m in models:
        arch = m.get("architecture") or {}
        pricing = m.get("pricing") or {}
        inputs = arch.get("input_modalities") or []
        outputs = arch.get("output_modalities") or ["text"]
        try:
            free = float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
        except (TypeError, ValueError):
            free = False
        if free and "image" in inputs and outputs == ["text"]:
            params = m.get("supported_parameters") or []
            out.append({
                "id": m["id"],
                "name": m.get("name") or m["id"],
                "context_length": m.get("context_length"),
                "json_mode": "response_format" in params or "structured_outputs" in params,
            })
    return sorted(out, key=lambda x: (not x["json_mode"], x["id"]))
