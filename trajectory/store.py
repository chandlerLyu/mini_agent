"""Trajectory persistence."""

from __future__ import annotations

import json
from pathlib import Path


class TrajectoryStore:
    def save(self, path: Path, state, config) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "info": {
                "status": state.exit_status,
                "final_answer": state.final_answer,
                "step_count": state.step_count,
                "model_calls": state.model_calls,
                "cost": state.total_cost,
            },
            "config": config.to_dict(),
            "state": state.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
