"""Small local HTTP client shared by Compass tools."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/systemone"


def ask(body: dict[str, Any], endpoint: str | None = None) -> tuple[dict[str, Any], float]:
    """Send one decision request and return the decoded response and wall time."""
    url = endpoint or os.environ.get("COMPASS_ENDPOINT", DEFAULT_ENDPOINT)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Compass returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Compass is unavailable at {url}: {error.reason}") from error
    return result, (time.perf_counter() - started) * 1000
