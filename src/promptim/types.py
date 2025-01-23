import copy
import json
from dataclasses import dataclass, field, fields
from typing import Callable, Optional, Any, Protocol
from uuid import UUID

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumps
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langsmith.schemas import Example, Run
from langsmith.utils import LangSmithConflictError
from pydantic import BaseModel, Field, model_validator
from promptim._utils import get_var_healer
import logging

logger = logging.getLogger(__name__)

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
    upload_to: str | None = field(
        default=None,
        metadata={
            "description": "Upload the prompt to the hub repository. Mutually exclusive with identifier."
        },
    )

    def __post_init__(self):
        if self.identifier and self.prompt_str:
            raise ValueError(
                "Cannot provide both identifier and prompt_str. Choose one."
            )
        elif not self.identifier and not self.prompt_str:
            raise ValueError("Must provide either identifier or prompt_str.")
        if self.identifier and not self.upload_to:
            self.upload_to = self.identifier


@dataclass(kw_only=True)
class PromptWrapper(PromptConfig):
    _cached: ChatPromptTemplate | None = None
    _postlude: RunnableBinding | BaseChatModel | None = None
    lineage: list["PromptWrapper"] | None = None
    extra: dict | None = None

    @classmethod
    def from_config(cls, config: PromptConfig | dict):
        if isinstance(config, dict):
            config = PromptConfig(**config)
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
                    new_model = None

                    if isinstance(bound_llm, RunnableBinding):
                        if tools := bound_llm.kwargs.get("tools"):
                            bound_llm.kwargs["tools"] = _ensure_stricty(tools)
                        if new_model:
                            bound_llm = new_model.bind(
                                **{
                                    k: v
                                    for k, v in bound_llm.kwargs.items()
                                    if k not in ("model", "model_name")
                                }
                            )
                    else:
                        if new_model:
                            bound_llm = new_model
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
                                    k: v
                                    for k, v in (
                                        dict((bound_llm.kwargs or {}))
                                        | (self.model_config or {})
                                    ).items()
                                    if k not in rebound_llm.kwargs
                                    and k not in ("model", "model_name")
                                }
                            ),
                            parser,
                        )
                    else:
                        postlude = bound_llm
                else:
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
            return (
                "{{messages}}"
                if isinstance(msg, MessagesPlaceholder)
                else msg.prompt.template
            )
        except Exception as e:
            raise NotImplementedError(
                f"Unsupported message template format. {msg}"
            ) from e

    def required_variables(self) -> set[str]:
        tmpl = self.load()
        return set(tmpl.messages[self.which].input_variables)

    def get_prompt_str_in_context(self, client: ls.Client | None = None) -> str:
        tmpl = self.load(client)
        formatted = []
        for i, msg in enumerate(tmpl.messages):
            kind = msg.__class__.__name__.replace("MessagePromptTemplate", "").replace(
                "Human", "User"
            )
            if i == self.which:
                tmpl = "{{messages}}" if isinstance(msg, MessagesPlaceholder) else msg.prompt.template  # type: ignore
                formatted.append(
                    f"""<TO_OPTIMIZE kind="{kind}">
{tmpl}
</TO_OPTIMIZE>"""
                )
            else:
                tmpl = "{{messages}}" if isinstance(msg, MessagesPlaceholder) else msg.prompt.template  # type: ignore
                formatted.append(
                    f"""<CONTEXT kind="{kind}">
{tmpl}
</CONTEXT>
"""
                )
        return "\n".join(formatted)

    @classmethod
    def from_prior(
        cls, prior: "PromptWrapper", output: str, extra_info: dict | None = None
    ):
        copied = prior._cached
        if not copied:
            raise ValueError("Cannot load from unloaded prior.")
        extra_info = extra_info or {}
        copied = copy.deepcopy(copied)
        tmpl = copied.messages[prior.which]
        tmpl.prompt.template = output  # type: ignore
        lineage = prior.lineage.copy() if prior.lineage else []
        lineage.append(prior)
        return cls(
            identifier=prior.identifier,
            prompt_str=prior.prompt_str,
            which=prior.which,
            _cached=copied,
            _postlude=prior._postlude,
            lineage=lineage,
            extra=extra_info,
            upload_to=prior.upload_to,
        )

    def push_prompt(
        self,
        *,
        include_model_info: bool = True,
        client: ls.Client | None = None,
    ) -> str:
        if not self.upload_to:
            raise ValueError("Cannot push prompt without an upload target.")
        client = client or ls.Client()
        prompt = self.load(client)
        identifier = self.upload_to.rsplit(":", maxsplit=1)[0]
        try:
            if not include_model_info or not self._postlude:
                new_id = client.push_prompt(identifier, object=prompt)
            else:
                seq = self._get_seq(client)
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

    def _get_seq(self, client: ls.Client | None = None):
        prompt = self.load(client)
        second = (
            self._postlude.first
            if isinstance(self._postlude, RunnableSequence)
            else self._postlude
        )
        if second:
            return RunnableSequence(prompt, second)
        return prompt

    @staticmethod
    def _push_seq(client: ls.Client, seq: RunnableSequence, identifier: str):
        manifest = json.loads(dumps(seq))
        manifest["id"] = ("langsmith", "playground", "PromptPlayground")
        return client.push_prompt(identifier, object=manifest)

    def dumps(self, push: bool = False) -> str:
        if push:
            identifier = self.push_prompt(include_model_info=False)
        else:
            identifier = self.identifier
        d = {
            "identifier": identifier,
            "prompt_str": (
                self.prompt_str if self.prompt_str else self.get_prompt_str_in_context()
            ),
            "model_config": self.model_config,
            "which": self.which,
            "manifest": self._get_seq(client=None),
        }
        return dumps(d)


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
            formatted = prompt.invoke(inputs)
            return await prompt_wrapper._postlude.ainvoke(formatted)

        return prompt_system

    @property
    def system_safe(self) -> SystemType:
        if self.system:
            return self.system

        prompt = PromptWrapper.from_config(self.initial_prompt)
        return self.get_prompt_system(prompt)


class OptimizedPromptOutput(Protocol):
    analysis: str
    hypothesis: str
    improved_prompt: str


def prompt_schema(
    og_prompt: PromptWrapper,
    schema: type[OptimizedPromptOutput] = OptimizedPromptOutput,
) -> type[OptimizedPromptOutput]:
    required_variables = og_prompt.required_variables()
    if required_variables:
        variables_str = ", ".join(f"{{{var}}}" for var in required_variables)
        prompt_description = (
            f" The prompt section being optimized contains the following f-string variables to be templated in: {variables_str}."
            " You must retain all of these variables in your improved prompt. No other input variables are allowed."
        )
    else:
        prompt_description = (
            " The prompt section being optimized contains no input f-string variables."
            " Any brackets {{ foo }} you emit will be escaped and not used."
        )

    pipeline = get_var_healer(set(required_variables), all_required=True)

    class OptimizedPromptOutput(BaseModel):
        """Schema for the optimized prompt output."""

        analysis: str = Field(
            description="First, analyze the current results and plan improvements to reconcile them."
        )
        hypothesis: str = Field(
            description="Second, write your hypothesis on what prompt intervention you are making to fix the prompt's errors."
        )
        improved_prompt: str = Field(
            description="The improved prompt text to replace the text contained within the"
            f" <TO_OPTIMIZE> and </TO_OPTIMIZE> tags, in f-string format. Do not includde <TO_OPTIMIZE> in your response. {prompt_description}"
        )

        @model_validator(mode="before")
        @classmethod
        def validate_input_variables(cls, data: Any) -> Any:
            assert "improved_prompt" in data
            data["improved_prompt"] = pipeline(data["improved_prompt"])
            return data

    return OptimizedPromptOutput


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
