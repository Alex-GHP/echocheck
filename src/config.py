from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Production configuration for training, evaluation, and inference."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths (all relative to PROJECT_ROOT unless absolute)
    raw_data_dir: Path = Path("data")
    processed_data_dir: Path = Path("processed_data")
    processed_data_jsonl_dir: Path = Path("processed_data_jsonl")
    model_output_dir: Path = Path("models/trainer_output")

    # Domain constants
    labels: tuple[str, ...] = ("center", "left", "right")

    # Data preprocessing
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_article_length: int = 50
    random_seed: int = 42
    jsonl_buffer_bytes: int = 8 * 1024 * 1024

    # Model
    base_model: str = "roberta-base"
    max_length: int = 512

    # Training (TrainingArguments)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 48
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 3
    logging_steps: int = 50
    metric_for_best_model: str = "eval_macro_f1"
    bf16: bool = True
    dataloader_num_workers: int = 0
    early_stopping_patience: int = 3
    run_name: str = "roberta-base-trainer-baseline"

    # Evaluation (test-set scoring)
    eval_model_dir: Path = Path("models/trainer_output/final")
    eval_test_data: Path = Path("processed_data_jsonl/test.jsonl")
    eval_output_file: Path = Path("evaluation_results.json")
    eval_batch_size: int = 48
    eval_run_name: str = "evaluation-test-set"

    # Weights & Biases
    wandb_project: str = Field(default="echocheck", alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, alias="WANDB_ENTITY")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")
    wandb_log_model: str = "end"
    wandb_watch: str = "gradients"

    # HuggingFace Hub
    hf_hub_repo_id: str = "alxdev/echocheck-political-stance"
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # Derived properties
    @computed_field
    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @computed_field
    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    @computed_field
    @property
    def label2id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    @computed_field
    @property
    def final_model_dir(self) -> Path:
        return self.model_output_dir / "final"


class SmokeSettings(Settings):
    """Tiny-scale overrides for the `--smoke` test."""

    raw_data_dir: Path = Path("/tmp/echocheck_smoke")
    processed_data_jsonl_dir: Path = Path("/tmp/echocheck_smoke")
    model_output_dir: Path = Path("/tmp/echocheck_smoke_output")

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    eval_steps: int = 10
    save_steps: int = 10
    save_total_limit: int = 1
    logging_steps: int = 2
    run_name: str = "smoke-test"


class SubsampleSettings(Settings):
    processed_data_jsonl_dir: Path = Path("processed_data_subsample_jsonl")
    model_output_dir: Path = Path("models/trainer_output_100k")

    eval_model_dir: Path = Path("models/trainer_output_100k/final")
    eval_test_data: Path = Path("processed_data_subsample_jsonl/test.jsonl")
    eval_output_file: Path = Path("evaluation_results_100k.json")

    run_name: str = "roberta-base-100k-baseline"
    eval_run_name: str = "evaluation-100k-baseline"


class WindowedSettings(SubsampleSettings):
    max_length: int = 256
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 96

    model_output_dir: Path = Path("models/trainer_output_windowed_100k")
    eval_model_dir: Path = Path("models/trainer_output_windowed_100k/final")
    eval_output_file: Path = Path("evaluation_results_windowed.json")

    run_name: str = "roberta-windowed-256ovl64-100k"
    eval_run_name: str = "evaluation-windowed-100k"

    window_size: int = 256
    stride: int = 64


class Windowed512Settings(WindowedSettings):
    max_length: int = 512
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 48

    window_size: int = 512

    model_output_dir: Path = Path("models/trainer_output_windowed512_100k")
    eval_model_dir: Path = Path("models/trainer_output_windowed512_100k/final")
    eval_output_file: Path = Path("evaluation_results_windowed512.json")

    run_name: str = "roberta-windowed-512ovl64-100k"
    eval_run_name: str = "evaluation-windowed512-100k"


settings = Settings()
