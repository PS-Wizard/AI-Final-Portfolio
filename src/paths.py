from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class VisionPaths:
    data: Path = DATASETS / "vision"
    processed: Path = OUTPUTS / "tables" / "vision" / "processed"
    figures: Path = OUTPUTS / "figures" / "vision"
    tables: Path = OUTPUTS / "tables" / "vision"
    histories: Path = OUTPUTS / "histories" / "vision"
    models: Path = OUTPUTS / "models" / "vision"
    summaries: Path = OUTPUTS / "model_summaries" / "vision"
    predictions: Path = OUTPUTS / "predictions" / "vision"


@dataclass(frozen=True)
class NlpPaths:
    data: Path = DATASETS / "nlp"
    processed: Path = OUTPUTS / "tables" / "nlp" / "processed"
    figures: Path = OUTPUTS / "figures" / "nlp"
    tables: Path = OUTPUTS / "tables" / "nlp"
    histories: Path = OUTPUTS / "histories" / "nlp"
    models: Path = OUTPUTS / "models" / "nlp"
    summaries: Path = OUTPUTS / "model_summaries" / "nlp"
    predictions: Path = OUTPUTS / "predictions" / "nlp"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


VISION = VisionPaths()
NLP = NlpPaths()
