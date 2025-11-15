from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
import pathlib
from loguru import logger

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML é necessário para carregar schemas YAML. Instale com `pip install pyyaml`.") from e


@dataclass(frozen=True)
class FeatureSchema:
    features: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    target: str = ""
    date_column: str = ""
    key_columns: List[str] = field(default_factory=list)
    frequency: Optional[str] = None  # e.g. "daily", "weekly", "monthly"
    dtypes: Dict[str, str] = field(default_factory=dict)


class SchemaLoader(Protocol):
    def load(self, path: str | pathlib.Path) -> FeatureSchema:
        ...


class YamlSchemaLoader:
    """Responsabilidade única: carregar um schema YAML e convertê-lo em FeatureSchema."""
    def load(self, path: str | pathlib.Path) -> FeatureSchema:
        logger.info(f"Loading schema from {path}...")
        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return FeatureSchema(
            features=raw.get("features", []),
            categorical=raw.get("categorical", []),
            target=raw.get("target", ""),
            date_column=raw.get("date_column", ""),
            key_columns=raw.get("key_columns", []),
            frequency=raw.get("frequency"),
            dtypes=raw.get("dtypes", {}) or {},
        )
