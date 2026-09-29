"""Minimal OpenAI-compatible chat client (image + text), built on `requests`.

No openai SDK dependency; talks to any /chat/completions endpoint that accepts
base64 image_url content parts.
"""
import base64
import mimetypes
import time
from typing import Optional

import requests

from . import config


def encode_image_data_url(image_path: str, max_pixels: Optional[int] = None) -> str:
    """Read an image file and return a base64 data URL.

    If ``max_pixels`` is set and the image exceeds it, the image is downscaled
    (aspect ratio preserved) before encoding. This is used only for endpoints
    with a hard pixel cap; GT boxes are
    normalized to 0-1000 so downscaling does not affect scoring. When no resize
    is needed, the original file bytes are sent unchanged.
    """
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"

    if max_pixels:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as im:
            w, h = im.size
            if w * h > max_pixels:
                import io
                # Target 95% of the cap so integer rounding never lands us back
                # over the limit.
                target = max_pixels * 0.95
                scale = (target / float(w * h)) ** 0.5
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                im = im.convert("RGB")
                im = im.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/jpeg;base64,{b64}"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
        temperature: float = config.DEFAULT_TEMPERATURE,
        top_p: float = config.DEFAULT_TOP_P,
        top_k: int = config.DEFAULT_TOP_K,
        timeout: int = config.DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = config.DEFAULT_MAX_RETRIES,
        max_retries_429: int = config.DEFAULT_MAX_RETRIES_429,
        max_image_pixels: Optional[int] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_429 = max_retries_429
        # Only set for endpoints with a hard pixel cap; None sends originals
         # unchanged, which is the default.
        self.max_image_pixels = max_image_pixels

    def complete(self, system_prompt: str, user_text: str, image_path: str) -> dict:
        """Single-turn call: system + (image, text) user.

        Returns {"content": str, "reasoning_content": str, "text": str} where
        `text` is the combined reasoning+content used for answer extraction (some
        reasoning models put the <answer> in reasoning_content rather than
        content).
        """
        data_url = encode_image_data_url(image_path, max_pixels=self.max_image_pixels)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # Sampling params pinned explicitly so all models compare equally.
        # Some endpoints reject temperature != 1; pass a negative value on the CLI
        # -> None here -> field omitted.
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Optional[Exception] = None
        attempt = 0          # counts general (non-429) attempts
        rl_attempt = 0       # counts rate-limit (429) attempts, separate budget
        while attempt < self.max_retries:
            try:
                resp = requests.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout
                )
                # 429 = rate limit: retry on its own budget with longer backoff,
                # without consuming the general retry count.
                if resp.status_code == 429:
                    last_err = RuntimeError(f"HTTP 429: {resp.text[:200]}")
                    rl_attempt += 1
                    if rl_attempt <= self.max_retries_429:
                        time.sleep(min(5 * rl_attempt, 60))
                        continue
                    raise last_err
                if resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                finish = choice.get("finish_reason")
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                # Combine for answer extraction: prefer content, but include
                # reasoning so an <answer> emitted only there is still found.
                text = content if content.strip() else reasoning
                if content.strip() and reasoning.strip() and "<answer>" not in content:
                    text = content + "\n" + reasoning
                # Empty response from content_filter/length is often transient at a
                # model's default (stochastic) temperature — retry a few times.
                if not text.strip() and finish in ("content_filter", "length") \
                        and attempt < self.max_retries - 1:
                    attempt += 1
                    time.sleep(min(2 ** attempt, 10))
                    continue
                return {
                    "content": content,
                    "reasoning_content": reasoning,
                    "text": text,
                    "finish_reason": finish,
                }
            except Exception as e:  # noqa: BLE001 - retry transient errors
                last_err = e
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"chat completion failed after {self.max_retries} tries: {last_err}")


class MockClient:
    """Offline stand-in used by --dry-run to exercise the full pipeline sans API."""

    def __init__(self, *args, **kwargs):
        self.model = "mock"

    def complete(self, system_prompt: str, user_text: str, image_path: str) -> dict:
        # Emit a plausibly-formatted answer for each family so parsers/scorers run.
        if "integer" in user_text:
            content = "<think>mock</think>\n<answer>1</answer>"
        elif "positive_points" in user_text:
            content = (
                "<think>mock</think>\n<answer>{\"boxes\":[100,100,400,400],"
                "\"positive_points\":[[200,200],[250,250],[300,300]],"
                "\"negative_points\":[[10,10],[20,20],[30,30]]}</answer>"
            )
        elif "LEFT panel" in user_text:
            content = "<think>mock</think>\n<answer>[[100,100,150,150],[200,200,260,260]]</answer>"
        else:
            content = "<think>mock</think>\n<answer>[100, 100, 400, 400]</answer>"
        return {"content": content, "reasoning_content": "", "text": content,
                "finish_reason": "stop"}
