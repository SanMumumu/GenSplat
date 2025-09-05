"""
Utilities for resolving checkpoint paths when using SwanLab instead of the previous tracker.

Supported URI schemes:
  - Local filesystem paths: returned as-is if they exist.
  - "swanlab://<project>/<exp_id>:<filename>": best-effort resolution to a local file.
      * In "local" or "offline" modes, SwanLab writes logs under SWANLAB_LOG_DIR (default: ./swanlog).
        This function will try to locate the file under that directory.
      * In "cloud" mode, SwanLab currently has no artifact download API. Raise an informative error.
  - (Back-compat) "previous-tracker://<...>": raise a clear error directing users to migrate.

This mirrors the previous "update_checkpoint_path" signature used across the codebase.
"""
from __future__ import annotations
from pathlib import Path
import os
from typing import Optional

import swanlab

def _resolve_local_swanlog(project: str, exp_name_or_id: str, filename: str) -> Optional[Path]:
    # Try default local swanlog folder
    log_root = Path(os.environ.get("SWANLAB_LOG_DIR", "swanlog"))
    # Heuristic search: project/exp_id or project/experiment_name directories
    candidates = [
        log_root / project / exp_name_or_id / filename,
        log_root / project / exp_name_or_id / "checkpoints" / filename,
        Path("checkpoints") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def update_checkpoint_path(path: Optional[str], swanlab_cfg: dict) -> Optional[str]:
    if path is None:
        return None

    # direct local path
    p = Path(path)
    if p.exists():
        return str(p)

    if path.startswith("swanlab://"):
        # swanlab://<exp_id_or_name>:<filename> OR swanlab://<project>/<exp_id_or_name>:<filename>
        payload = path[len("swanlab://"):]
        if "/" in payload:
            project, rest = payload.split("/", 1)
        else:
            # fall back to configured project
            project = swanlab_cfg.get("project", None) or os.environ.get("SWANLAB_PROJ_NAME", "")
            rest = payload
        parts = rest.split(":")
        if len(parts) == 1:
            exp = parts[0]
            filename = "model.ckpt"
        elif len(parts) == 2:
            exp, filename = parts
        else:
            raise ValueError("Invalid swanlab URI; expected swanlab://<project>/<exp_id_or_name>:<filename>")

        # In local/offline mode, attempt local resolution
        mode = swanlab_cfg.get("mode", os.environ.get("SWANLAB_MODE", "cloud"))
        if mode in ("local", "offline"):
            resolved = _resolve_local_swanlog(project, exp, filename)
            if resolved is None:
                raise FileNotFoundError(f"Could not find {filename} under local SwanLab logdir for project={project}, exp={exp}. "
                                        f"Checked SWANLAB_LOG_DIR={os.environ.get('SWANLAB_LOG_DIR','swanlog')}.")
            return str(resolved)

        # Cloud mode: no artifact download API yet
        raise NotImplementedError("SwanLab cloud mode currently has no artifact download API for files. "
                                  "Please provide a local filesystem path to the checkpoint, or run in offline/local mode so the file exists locally.")

    if path.startswith("previous-tracker://"):
        raise NotImplementedError("the previous tracker artifact URIs are no longer supported. Migrate to SwanLab or provide a local path.")

    # unknown scheme
    return path
