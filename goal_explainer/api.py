"""
Meta Model API client with disk caching.

Docs: https://dev.meta.ai/docs/ — base URL ``https://api.meta.ai/v1``.

- SAM 3.1: ``POST /responses`` (``model="sam-3.1"``), one noun-phrase prompt per
  request, streamed SSE whose text has one line per frame (see ``parse_sam_text``).
- Muse Spark: ``POST /chat/completions`` (OpenAI-compatible) with
  ``response_format`` JSON schema.
- Videos are uploaded once via ``POST /files`` (``purpose=user_data``).

Cache: SHA256(request payload) → JSON on disk in ``runs/<clip_id>/.cache/``.
The payload always contains the media (data URL or file id derived from the
file hash), so different frames never share a cache entry.

Environment: ``MODEL_API_KEY`` (or ``META_API_KEY``), optional ``META_API_BASE``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os

import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

SAM_MODEL: str = "sam-3.1"
SPARK_MODEL: str = "muse-spark-1.3"

_DEFAULT_BASE_URL = "https://api.meta.ai/v1"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load ``KEY=value`` lines from *path*; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))


load_dotenv()
_TIMEOUT_S = 300.0
_MAX_RETRIES = 4
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class MetaAPIClient:
    """Synchronous client for the Meta Model API with transparent disk cache."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.base_url: str = (
            base_url or os.environ.get("META_API_BASE", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.api_key: str = (
            api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY", "")
        )
        self.cache_dir: Path | None = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(_TIMEOUT_S),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    # ------------------------------------------------------------------
    # Public calls
    # ------------------------------------------------------------------

    def upload_file(self, path: Path) -> str:
        """Upload a file (e.g. the clip) once and return its ``file_id``."""
        path = Path(path)
        key = self._cache_key({"upload": _sha256_file(path)})
        cached = self._read_cache(key)
        if cached is not None:
            return cached["id"]
        with open(path, "rb") as fh:
            resp = self._send(
                "POST", "/files",
                data={"purpose": "user_data"},
                files={"file": (path.name, fh, "video/mp4")},
            )
        result = resp.json()
        self._write_cache(key, result)
        return result["id"]

    def sam(
        self,
        prompt: str,
        *,
        video_file_id: str | None = None,
        image_jpeg: bytes | None = None,
    ) -> str:
        """Run SAM 3.1 with one noun-phrase prompt; return the raw streamed text."""
        if (video_file_id is None) == (image_jpeg is None):
            raise ValueError("Pass exactly one of video_file_id or image_jpeg.")
        media: dict[str, Any] = (
            {"type": "input_video", "file_id": video_file_id}
            if video_file_id is not None
            else {"type": "input_image", "image_url": _data_url(image_jpeg)}
        )
        payload = {
            "model": SAM_MODEL,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, media],
            }],
            "stream": True,
            # Boxes are all we use; one_bit keeps the stream small.
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        key = self._cache_key(payload)
        cached = self._read_cache(key)
        if cached is not None:
            return cached["text"]
        text = self._stream_text("/responses", payload)
        self._write_cache(key, {"text": text})
        return text

    def spark_video_json(self, prompt: str, video_file_id: str, model: str = SPARK_MODEL) -> dict[str, Any]:
        """Ask Muse Spark about an uploaded video; the answer must be one JSON object.

        Uses the Responses API (``input_video`` + ``file_id``); its structured-output
        shape is not documented, so the prompt asks for JSON and we parse it.
        """
        payload = {
            "model": model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt},
                            {"type": "input_video", "file_id": video_file_id}],
            }],
        }
        key = self._cache_key(payload)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        body = self._send("POST", "/responses", json=payload).json()
        text = "".join(c.get("text", "") for item in body.get("output", []) for c in item.get("content", []) or []
                       if c.get("type") == "output_text")
        result = json.loads(text[text.index("{"): text.rindex("}") + 1])
        self._write_cache(key, result)
        return result

    def spark_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        images: list[bytes] | None = None,
        model: str = SPARK_MODEL,
    ) -> dict[str, Any]:
        """Ask Muse Spark for a JSON object matching *schema* (images are JPEG bytes)."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            },
        }
        key = self._cache_key(payload)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        resp = self._send("POST", "/chat/completions", json=payload)
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        self._write_cache(key, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, key: str, data: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Request with retry on 429/5xx/network errors; other 4xx raise at once."""
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                logger.warning("%s %s failed (%s), retrying", method, url, exc)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning("%s %s → %d, retrying in %.0fs", method, url, resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.is_error:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} from {url}: {resp.text[:500]}",
                    request=resp.request, response=resp,
                )
            return resp
        raise AssertionError("unreachable")

    def _stream_text(self, url: str, payload: dict[str, Any]) -> str:
        """POST with ``stream: true`` and join the ``output_text`` deltas."""
        for attempt in range(_MAX_RETRIES):
            parts: list[str] = []
            try:
                with self._http.stream("POST", url, json=payload) as resp:
                    if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES - 1:
                        time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                        continue
                    if resp.is_error:
                        resp.read()
                        raise httpx.HTTPStatusError(
                            f"{resp.status_code} from {url}: {resp.text[:500]}",
                            request=resp.request, response=resp,
                        )
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        event = json.loads(data)
                        etype = event.get("type", "")
                        if etype == "response.output_text.delta":
                            parts.append(event.get("delta", ""))
                        elif etype in ("error", "response.failed"):
                            raise RuntimeError(f"SAM stream error: {data[:500]}")
                        elif etype == "response.completed":
                            break
                return "".join(parts)
            except httpx.TransportError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{url}: retries exhausted")


# ---------------------------------------------------------------------------
# SAM output parsing (official parser: mask payloads are base85 and may hold any
# character, so a hand-written regex is not safe)
# ---------------------------------------------------------------------------

def parse_sam_text(text: str) -> list[dict[str, Any]]:
    """Parse SAM's streamed text into ``[{frame, id, bbox, conf, mask}]``.

    ``mask`` is a :class:`meta_sam_parser.SegmentationMask` (box-local) or None.
    Frames may be skipped or out of order; ids are stable per track.
    """
    from meta_sam_parser import video_segmentation_format

    snaps = video_segmentation_format().create_parser().push(text + "\n")
    records = snaps[-1].records if snaps else ()
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for r in records:
        if r.kind not in ("box", "mask"):
            continue
        frame = r.frame.frame_index if r.frame is not None else 0
        d = out.setdefault((frame, r.object_id), {
            "frame": frame, "id": int(r.object_id), "bbox": None, "conf": 1.0, "mask": None,
        })
        if r.confidence is not None:
            d["conf"] = float(r.confidence)
        if r.kind == "box":
            d["bbox"] = [int(r.left), int(r.top), int(r.right), int(r.bottom)]
        else:
            d["mask"] = r.mask
    return [d for d in out.values() if d["bbox"] is not None]


def mask_to_frame(det: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """Full-frame uint8 mask (0/1) for a detection whose mask is box-local."""
    from meta_sam_parser import decode_mask_to_raster

    full = np.zeros(shape[:2], np.uint8)
    m = det.get("mask")
    if m is None:
        return full
    local = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    x1, y1 = det["bbox"][0], det["bbox"][1]
    h = min(m.height, shape[0] - y1)
    w = min(m.width, shape[1] - x1)
    full[y1:y1 + h, x1:x1 + w] = local[:h, :w]
    return full


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_client(clip_id: str) -> MetaAPIClient:
    """Client whose disk cache lives under ``runs/<clip_id>/.cache/``."""
    return MetaAPIClient(cache_dir=Path("runs") / clip_id / ".cache")
