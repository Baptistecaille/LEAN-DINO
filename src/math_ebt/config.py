"""Typed configuration. One dataclass per YAML block; no dict juggling downstream."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectCfg:
    name: str = "math_ebt"
    run_name: str = "dino_v0"
    seed: int = 42
    output_dir: str = "outputs/dino_v0"


@dataclass
class DataCfg:
    raw_dir: str = "data/raw"
    splits_dir: str = "data/splits"
    tokenizer_path: str = "data/tokenizer/bpe.json"
    max_seq_len: int = 512
    max_seq_len_local: int = 256
    num_local_views: int = 4
    num_workers: int = 8
    view_mode: str = "certified"  # certified | naive | dropout_only


@dataclass
class ModelCfg:
    vocab_size: int = 32000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1


@dataclass
class DinoCfg:
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    out_dim: int = 4096
    student_temp: float = 0.10
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_steps: int = 10000
    center_momentum: float = 0.9
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    freeze_last_layer_steps: int = 2000


@dataclass
class TrainingCfg:
    batch_size: int = 32
    grad_accum_steps: int = 8
    total_steps: int = 100000
    warmup_steps: int = 10000
    lr: float = 5e-4
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    grad_clip: float = 1.0
    precision: str = "bf16"
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000


@dataclass
class EvalCfg:
    probe_num_classes: int = 15
    probe_epochs: int = 20
    probe_lr: float = 1e-3
    retrieval_top_k: list[int] = field(default_factory=lambda: [10, 100])


@dataclass
class GatesCfg:
    cos_pos_min: float = 0.80
    cos_rand_max: float = 0.30
    top10_jaccard_min: float = 0.70
    probe_top1_min: float = 0.55
    min_stable_steps: int = 50000


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    dino: DinoCfg = field(default_factory=DinoCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    gates: GatesCfg = field(default_factory=GatesCfg)

    @property
    def effective_batch(self) -> int:
        return self.training.batch_size * self.training.grad_accum_steps

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return Config(
            project=ProjectCfg(**raw.get("project", {})),
            data=DataCfg(**raw.get("data", {})),
            model=ModelCfg(**raw.get("model", {})),
            dino=DinoCfg(**raw.get("dino", {})),
            training=TrainingCfg(**raw.get("training", {})),
            eval=EvalCfg(**raw.get("eval", {})),
            gates=GatesCfg(**raw.get("gates", {})),
        )
