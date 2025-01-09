"""Task definition for health classification (10-class)."""

from typing import List, Callable
import json
from pathlib import Path

from langsmith.schemas import Run, Example
from krishpromptim.prompt_types import Task, Dataset
from .evaluators import evaluators as health_ten_class_evaluators


class HealthClassificationTask10(Task):
    """Task for classifying health conditions into ten categories."""

    def __init__(self, custom_evaluators: List[Callable[[Run, Example], dict]] = None):
        """Initialize the task with optional custom evaluators.

        Args:
            custom_evaluators: Optional list of custom evaluator functions. If provided,
                             these will replace the default evaluator.
        """
        evaluators = (
            custom_evaluators
            if custom_evaluators is not None
            else health_ten_class_evaluators
        )

        # Load config from json
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        super().__init__(
            name=config["name"],
            description=config["description"],
            dataset=Dataset(**config["dataset"]),
            evaluators=evaluators,
            evaluator_descriptions=config["evaluator_descriptions"],
            initial_prompt=config["initial_prompt"],
            optimizer=config["optimizer"],
        )


# Export the task class
__all__ = ["HealthClassificationTask10"]
