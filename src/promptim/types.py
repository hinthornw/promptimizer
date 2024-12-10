import copy
import json
from dataclasses import dataclass, field, fields
from typing import Callable, Optional
from uuid import UUID

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumps
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langsmith.schemas import Example, Run
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel, Field

DEFAULT_PROMPT_MODEL_CONFIG = {"model": "claude-3-5-haiku-20241022"}
DEFAULT_OPTIMIZER_MODEL_CONFIG = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens_to_sample": 8192,
}


SystemType = Callable[[ChatPromptTemplate, dict], dict]
"""Takes the current prompt and the example inputs and returns the results."""


@dataclass(kw_only=True)
class PromptConfig:
    identifier: str | None = field(
        default=None,
        metadata={
            "description": "Identifier for a prompt from the hub repository. Mutually exclusive with prompt_str."
        },
    )
    prompt_str: str | None = field(
        default=None,
        metadata={
            "description": "Raw prompt string to optimize locally. Mutually exclusive with identifier."
        },
    )
    model_config: dict | None = field(
        default=None,
        metadata={
            "description": "Configuration dictionary specifying model parameters for optimization."
        },
    )
    which: int = field(
        default=0,
        metadata={"description": "Index of the message to optimize within the prompt."},
    )

    def __post_init__(self):
        if self.identifier and self.prompt_str:
            raise ValueError(
                "Cannot provide both identifier and prompt_str. Choose one."
            )
        elif not self.identifier and not self.prompt_str:
            raise ValueError("Must provide either identifier or prompt_str.")


@dataclass(kw_only=True)
class PromptWrapper(PromptConfig):
    _cached: ChatPromptTemplate | None = None
    _postlude: RunnableBinding | BaseChatModel | None = None

    @classmethod
    def from_config(cls, config: PromptConfig):
        return cls(
            identifier=config.identifier,
            prompt_str=config.prompt_str,
            model_config=config.model_config,
            which=config.which,
        )

    def load(self, client: ls.Client | None = None) -> ChatPromptTemplate:
        if self._cached is None:
            if self.prompt_str:
                self._cached = ChatPromptTemplate.from_messages(
                    [("user", self.prompt_str)]
                )
                self._postlude = init_chat_model(
                    **(self.model_config or DEFAULT_PROMPT_MODEL_CONFIG)
                )
            else:
                client = client or ls.Client()
                postlude = None
                prompt = client.pull_prompt(self.identifier, include_model=True)
                if isinstance(prompt, RunnableSequence):
                    prompt, bound_llm = prompt.first, prompt.steps[1]
                    if isinstance(bound_llm, RunnableBinding):
                        if tools := bound_llm.kwargs.get("tools"):
                            bound_llm.kwargs["tools"] = _ensure_stricty(tools)
                    if isinstance(prompt, StructuredPrompt) and isinstance(
                        bound_llm, RunnableBinding
                    ):
                        seq: RunnableSequence = prompt | bound_llm.bound

                        rebound_llm = seq.steps[1]
                        if tools := rebound_llm.kwargs.get("tools"):
                            rebound_llm.kwargs["tools"] = _ensure_stricty(tools)
                        parser = seq.steps[2]
                        postlude = RunnableSequence(
                            rebound_llm.bind(
                                **{
                                    **{
                                        k: v
                                        for k, v in (bound_llm.kwargs or {}).items()
                                        if k not in rebound_llm.kwargs
                                    },
                                    **(self.model_config or {}),
                                }
                            ),
                            parser,
                        )
                    else:
                        postlude = bound_llm
                else:
                    # Default to gpt-4o-mini
                    postlude = init_chat_model(
                        **(self.model_config or DEFAULT_PROMPT_MODEL_CONFIG)
                    )
                    if isinstance(prompt, StructuredPrompt):
                        postlude = RunnableSequence(*(prompt | postlude).steps[1:])
                self._cached = prompt
                self._postlude = postlude
        return self._cached

    def get_prompt_str(self, client: ls.Client | None = None) -> str:
        tmpl = self.load(client)
        msg = tmpl.messages[self.which]
        try:
            return msg.prompt.template  # type: ignore
        except Exception as e:
            raise NotImplementedError(
                f"Unsupported message template format. {msg}"
            ) from e

    def get_prompt_str_in_context(self, client: ls.Client | None = None) -> str:
        tmpl = self.load(client)
        formatted = []
        for i, msg in enumerate(tmpl.messages):
            kind = msg.__class__.__name__.replace("MessagePromptTemplate", "").replace(
                "Human", "User"
            )
            if i == self.which:
                formatted.append(
                    f"""<TO_OPTIMIZE kind="{kind}">
{msg.prompt.template}
</TO_OPTIMIZE>"""
                )
            else:
                formatted.append(
                    f"""<CONTEXT kind="{kind}">
{msg.prompt.template}
</CONTEXT>
"""
                )
        return "\n".join(formatted)

    @classmethod
    def from_prior(cls, prior: "PromptWrapper", output: str):
        copied = prior._cached
        if not copied:
            raise ValueError("Cannot load from unloaded prior.")
        copied = copy.deepcopy(copied)
        tmpl = copied.messages[prior.which]
        tmpl.prompt.template = output  # type: ignore
        return cls(
            identifier=prior.identifier,
            prompt_str=prior.prompt_str,
            which=prior.which,
            _cached=copied,
            _postlude=prior._postlude,
        )

    def push_prompt(
        self,
        *,
        identifier: Optional[str] = None,
        include_model_info: bool = True,
        client: ls.Client,
    ) -> str:
        prompt = self.load(client)
        identifier = identifier or self.identifier.rsplit(":", maxsplit=1)[0]
        try:
            if not include_model_info or not self._postlude:
                new_id = client.push_prompt(identifier, object=prompt)
            else:
                second = (
                    self._postlude.first
                    if isinstance(self._postlude, RunnableSequence)
                    else self._postlude
                )
                seq = RunnableSequence(prompt, second)
                return self._push_seq(client, seq, identifier)

        except LangSmithConflictError:
            return identifier

        return ":".join(
            new_id
            # Remove the https:// prefix
            .split("/prompts/", maxsplit=1)[1]
            # Rm query string
            .split("?")[0]
            # Split the repo from the commit hash
            .rsplit("/", maxsplit=1)
        )

    @staticmethod
    def _push_seq(client: ls.Client, seq: RunnableSequence, identifier: str):
        manifest = json.loads(dumps(seq))
        manifest["id"] = ("langsmith", "playground", "PromptPlayground")
        return client.push_prompt(identifier, object=manifest)


@dataclass(kw_only=True)
class TaskLike:
    """Represents a specific task for prompt optimization."""

    name: str
    """The identifier for the task, used for logging and referencing."""
    dataset: str
    """The name of the dataset in LangSmith to be used for training and evaluation."""
    initial_prompt: PromptConfig
    """The starting prompt configuration, which will be optimized during the process."""
    description: str = ""
    """A detailed explanation of the task's objectives and constraints."""
    evaluator_descriptions: dict = field(default_factory=dict)
    """A mapping of evaluator names to their descriptions, used to guide the optimization process."""
    baseline_experiment: Optional[UUID] = None
    """The UUID of a previous experiment to use as a baseline for comparison, if available."""


@dataclass(kw_only=True)
class Task(TaskLike):
    """Represents a specific task for prompt optimization with additional execution details."""

    evaluators: list[Callable[[Run, Example], dict]]
    """A list of functions that assess the quality of model outputs, each returning a score and optional feedback."""
    system: Optional[SystemType] = None
    """A custom system configuration for executing the prompt, allowing for task-specific processing."""

    @classmethod
    def from_dict(cls, d: dict):
        d_ = d.copy()
        kwargs = {"initial_prompt": PromptWrapper(**d_.pop("initial_prompt")), **d_}

        field_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        return cls(**kwargs)

    def describe(self):
        descript = self.description if self.description else self.name
        evaluator_desc = "\n".join(
            [f"- {key}: {value}" for key, value in self.evaluator_descriptions.items()]
        )
        return f"{descript}\n\nDescription of scores:\n{evaluator_desc}"

    @staticmethod
    def get_prompt_system(prompt_wrapper: PromptWrapper):
        async def prompt_system(prompt: ChatPromptTemplate, inputs: dict):
            return await prompt_wrapper._postlude.ainvoke(prompt.invoke(inputs))

        return prompt_system

    @property
    def system_safe(self) -> SystemType:
        if self.system:
            return self.system

        prompt = PromptWrapper.from_config(self.initial_prompt)
        return self.get_prompt_system(prompt)


class OptimizedPromptOutput(BaseModel):
    """Schema for the optimized prompt output."""

    analysis: str = Field(
        description="First, analyze the current results and plan improvements to reconcile them."
    )
    improved_prompt: str = Field(description="The improved prompt text")


def _ensure_stricty(tools: list) -> list:
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            strict = None
            if func := tool.get("function"):
                if parameters := func.get("parameters"):
                    if "strict" in parameters:
                        strict = parameters["strict"]
            if strict is not None:
                tool = copy.deepcopy(tool)
                tool["function"]["strict"] = strict
        result.append(tool)
    return result
