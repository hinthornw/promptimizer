from abc import ABC, abstractmethod
from typing import List, Union, Optional, TypeVar, Generic
from dataclasses import dataclass

from promptim import types as pm_types
from promptim.trainer import PromptTrainer


@dataclass
class AlgorithmConfig:
    """Base configuration for training algorithms."""

    train_size: Optional[int] = None
    batch_size: int = 40
    epochs: int = 1
    debug: bool = False


C = TypeVar("C", bound=AlgorithmConfig)


class BaseAlgorithm(Generic[C], ABC):
    """
    Abstract base that defines the macro-level training loop
    or search procedure (epochs, phases, etc.).
    """

    config_cls = AlgorithmConfig

    def __init__(self, config: Optional[Union[dict, AlgorithmConfig]] = None):
        self.config = self._resolve_config(config or {})

    def _resolve_config(self, config: Union[dict, AlgorithmConfig]) -> AlgorithmConfig:
        if isinstance(config, dict):
            return self.config_cls(**config)
        return config

    @abstractmethod
    async def run(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        initial_population: Union[pm_types.PromptWrapper, List[pm_types.PromptWrapper]],
        train_examples: list[pm_types.Example],
        dev_examples: list[pm_types.Example],
        *,
        system_config: Optional[dict] = None,
        annotation_queue: Optional[str] = None,
        commit_prompts: bool = False,
        experiment_name: str = "Prompt Optimization",
        baseline_scores: Optional[dict] = None,
        baseline_experiment_results: Optional[list] = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        """
        Execute the training/evolution procedure using the trainer's capabilities.

        Args:
            trainer: The PromptTrainer instance providing evaluation and utilities
            task: The task to optimize for
            initial_population: Single prompt or list of prompts to start with
            system_config: Optional system-level configuration
            annotation_queue: Optional queue for manual review
            commit_prompts: Whether to commit prompts to LangSmith
            experiment_name: Optional name for the experiment

        Returns:
            Tuple of (best prompt, best score)
        """
        pass
