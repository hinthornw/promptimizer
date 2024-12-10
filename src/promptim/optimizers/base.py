from abc import ABC, abstractmethod
from typing import List
from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import types as pm_types


class BaseOptimizationAlgorithm(ABC):
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
