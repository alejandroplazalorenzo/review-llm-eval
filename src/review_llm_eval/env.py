"""Record the environment of a run: Ollama version, model digest, server settings,
other models loaded, GPU. Every experiment stores this next to its numbers."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from review_llm_eval.client import OllamaClient
from review_llm_eval.config import EXPECTED_DIGESTS, SERVER_ENV_VARS


def server_settings() -> dict[str, str]:
    """The server's effective settings. The Windows app logs them at start-up ("server
    config" line of server.log); that is the truth for the running server, unlike this
    process's environment. Falls back to this process's environment variables."""
    log = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
    if log.is_file():
        lines = [ln for ln in _read_lines(log) if "server config" in ln]
        if lines:
            found = {
                var: m.group(1)
                for var in SERVER_ENV_VARS
                if (m := re.search(rf"{var}:(\S*)", lines[-1]))
            }
            found["source"] = "server.log"
            return found
    values = {var: os.environ.get(var, "unset") for var in SERVER_ENV_VARS}
    values["source"] = "client environment"
    return values


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _nvidia_smi(fields: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def gpu_name() -> str:
    return _nvidia_smi("name,memory.total")


def gpu_state() -> str:
    """Temperature, SM clock and memory in use: a laptop GPU throttles under long loads,
    so speed numbers are only comparable when these are recorded too."""
    return _nvidia_smi("temperature.gpu,clocks.sm,memory.used")


def model_info(client: OllamaClient, model: str) -> dict[str, Any]:
    for tag in client.tags():
        if tag.get("name") == model:
            digest = str(tag.get("digest", ""))
            return {
                "model": model,
                "digest": digest[:12],
                "digest_matches_readme": digest.startswith(EXPECTED_DIGESTS.get(model, "?")),
                "quantization": tag.get("details", {}).get("quantization_level"),
                "parameter_size": tag.get("details", {}).get("parameter_size"),
            }
    return {"model": model, "digest": None, "digest_matches_readme": False}


def loaded_models(client: OllamaClient) -> list[dict[str, Any]]:
    return [
        {
            "name": m.get("name"),
            "size_vram": m.get("size_vram"),
            "size": m.get("size"),
            "context_length": m.get("context_length"),
        }
        for m in client.loaded()
    ]


def collect(client: OllamaClient, model: str) -> dict[str, Any]:
    return {
        "ollama_version": client.version(),
        "ollama_url": client.base_url,
        "server": server_settings(),
        "model": model_info(client, model),
        "loaded_before": loaded_models(client),
        "gpu": gpu_name(),
        "gpu_state_before": gpu_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
