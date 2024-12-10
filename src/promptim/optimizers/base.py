from abc import ABC, abstractmethod
from typing import List, Type
from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import types as pm_types
from dataclasses import dataclass, field, is_dataclass, asdict
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model

MODEL_TYPE = str | BaseChatModel | dict


@dataclass(kw_only=True)
class Config:
    kind: str
    model: MODEL_TYPE = field(
        default_factory=lambda: {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens_to_sample": 8192,
        }
    )


class BaseOptimizer(ABC):
    config_cls: Type[Config]

    def __init__(self, *, model: MODEL_TYPE):
        self.model = _resolve_model(model)

    @classmethod
    def from_config(cls, config: dict | Config):
        if is_dataclass(config):
            config = asdict(config)
        config_ = {k: v for k, v in config.items() if k != "kind"}
        return cls(**config_)

    @abstractmethod
    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: List[pm_types.PromptWrapper],
    ) -> pm_types.PromptWrapper:
        """Given the current prompt and the latest evaluation results,
        propose a new and improved prompt variant."""

    def on_epoch_start(self, epoch: int, task: pm_types.Task):
        """Hook for any setup needed at the start of each epoch."""


# Private utils


def _resolve_model(model: MODEL_TYPE) -> dict:
    if isinstance(model, dict):
        return init_chat_model(**model)
    elif isinstance(model, BaseChatModel):
        return model
    else:
        return init_chat_model(model=model)
