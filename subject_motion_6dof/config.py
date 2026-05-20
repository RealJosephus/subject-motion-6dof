from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class CfgNode(dict):
    """Small attribute-access config node compatible with the vendored modules."""

    def __init__(self, value: dict[str, Any] | None = None):
        super().__init__()
        for key, item in (value or {}).items():
            self[key] = self._wrap(item)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def clone(self) -> "CfgNode":
        return CfgNode(deepcopy(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for key, value in self.items():
            if isinstance(value, CfgNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, CfgNode) else v for v in value]
            else:
                out[key] = value
        return out


def load_config(path: str | Path) -> CfgNode:
    with open(path, "r", encoding="utf-8") as f:
        return CfgNode(yaml.safe_load(f))


def save_config(cfg: CfgNode, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
