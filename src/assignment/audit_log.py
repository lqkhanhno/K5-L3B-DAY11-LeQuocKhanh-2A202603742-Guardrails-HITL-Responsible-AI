"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + timestamp bắt đầu (theo request_id để khớp với output)."""
        request_id = request_id or f"req-{len(self.logs) + 1}-{utc_now_iso()}"
        self._open[request_id] = datetime.now(timezone.utc).timestamp()
        self.logs.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "input": text,
                "timestamp": utc_now_iso(),
                "output": None,
                "blocked": None,
                "layer": None,
                "latency_ms": None,
            }
        )
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Ghi output + quyết định layer + latency; gắn vào entry tương ứng."""
        now_iso = utc_now_iso()
        entry: dict | None = None

        # Tìm entry mở tương ứng (theo request_id, nếu có)
        if request_id is not None:
            for log in reversed(self.logs):
                if log.get("request_id") == request_id:
                    entry = log
                    break

        started = self._open.pop(request_id, None) if request_id else None
        if started is None and entry is None:
            # Không có input record khớp → tạo entry mới (output-only)
            started = datetime.now(timezone.utc).timestamp()
            entry = {
                "request_id": request_id or f"req-{len(self.logs) + 1}",
                "user_id": user_id,
                "input": None,
                "timestamp": now_iso,
                "output": None,
                "blocked": None,
                "layer": None,
                "latency_ms": None,
            }
            self.logs.append(entry)

        if entry is not None:
            latency_ms = round((datetime.now(timezone.utc).timestamp() - started) * 1000, 2) if started is not None else None
            entry.update(
                {
                    "output": text,
                    "blocked": blocked,
                    "layer": layer,
                    "latency_ms": latency_ms,
                    "output_timestamp": now_iso,
                }
            )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Audit log exported → {path}")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()