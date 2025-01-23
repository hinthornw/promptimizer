from typing import List, Sequence
from langsmith.evaluation._arunner import ExperimentResultRow
from langchain_core.messages import AIMessage
from dataclasses import dataclass, field
from promptim import types as pm_types
from promptim import _utils as pm_utils
from promptim.optimizers import base as optimizers
from typing_extensions import Literal
from trustcall import create_extractor
import langsmith as ls
import html


DEFAULT_METAPROMPT = """Diagnose and optimize the quality of the prompt over the target task. Understand the underlying model's behavior patterns, and the underlying data generating process
so you know how to make the right improvements. Understand the prompt only has the individual input context. Use the aggregate results for deeper understanding.

## Current prompt

The following is the current best-performing prompt:
{current_hypo}
<current_prompt>
{current_prompt}
</current_prompt>

Your generations will replace the content within the <TO_OPTIMIZE></TO_OPTIMIZE> tags. The rest is fixed context over which you have no control. The TO_OPTIMIZE and CONTEXT\
 tags are provided here to help you disambiguateand not present in the prompt itself.

## Previous Prompt Attempts

You previously attempted to use the following prompts, but they earned worse scores than the current one:
<other_attempts>
{other_attempts}
</other_attempts>

Think about what hypotheses you were testing in these previous attempts. None of them were optimal. Think through why to explore better options and better understand the underlying domain.

## Annotated results:
The prompt sees the input variables. It produces the outputs.
The reference is hidden to the prompt and represents the expectations of the task.
<results>
{annotated_results}
</results>

## Task description:
<task_description>
{task_description}
</task_description>

Unless otherwise specified, higher scores are better (try to maximize scores). Aim for perfect scores across all examples.

In your head, search through all edits, planning the optimization step-by-step:
1. Analyze the current results and where they fall short
2. Identify patterns in the underlying data generating process for the dataset.
3. Identify patterns in successful vs unsuccessful cases.
4. Generate hypotheses about what would help fix the shortcomings of the existing prompt.Propose specific improvements to address the shortcomings. Improvements cannot mention reference outputs, which are unavailable to the model.
5. Generate an improved prompt based on the most promising hypothesis.

The improved prompt must keep all original input variables.
Focus on targeted improvements rather than re-hashing what the prompt already handles well.

Use prompting strategies as appropriate for the task. For logic and math, consider encourage more chain-of-thought reasoning, 
or include reasoning trajectories to induce better performance. For creative tasks, consider adding style guidelines.
Or consider including synthetic exemplars. Take all the time you need, but DO NOT REPEAT HYPOTHESES FROM PREVIOUS ATTEMPTS. Update your priors by thinking why they're disproven, then try something new."""


@dataclass(kw_only=True)
class Config(optimizers.Config):
    """Configuration for the metaprompt optimization algorithm."""

    kind: Literal["metaprompt"] = field(
        default="metaprompt",
        metadata={
            "description": "The meta-prompt optimizer that uses an LLM to analyze and improve prompts."
        },
    )
    meta_prompt: str = field(
        default=DEFAULT_METAPROMPT,
        metadata={
            "description": "The meta-prompt to use for analyzing and improving prompts."
        },
    )


class MetaPromptOptimizer(optimizers.BaseOptimizer):
    """
    This is the original style meta-prompt algorithm:
    It takes the current results and uses the meta-prompt to propose a new prompt.
    """

    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        max_reasoning_steps: int = 5,
        meta_prompt: str | None = None,
    ):
        super().__init__(model=model)
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT
        self.max_reasoning_steps = max_reasoning_steps

    @ls.traceable(run_type="prompt", name="meta_prompt")
    def format(self, **kwargs):
        return self.meta_prompt.format(**kwargs)

    def _format_results(self, results: List[ExperimentResultRow]) -> str:
        formatted = []
        for i, r in enumerate(results):
            formatted.append(f"Example {i+1}:")
            formatted.append(f'Input: {r["run"].inputs}')
            if r["example"].outputs:
                formatted.append(
                    f'Reference (hidden from prompt): {r["example"].outputs}'
                )
            formatted.append(f'Prompt output: {r["run"].outputs}')
            formatted.append("Evaluations:")
            for eval_result in r["evaluation_results"]["results"]:
                formatted.append(f"- {eval_result.key}: {eval_result.score}")
                if eval_result.comment:
                    formatted.append(f"  Comment: {eval_result.comment}")
            formatted.append("---")
        return "\n".join(formatted)

    @ls.traceable
    async def improve_prompt(
        self,
        history: Sequence[Sequence[pm_types.PromptWrapper]],
        results: List[ExperimentResultRow],
        task: pm_types.Task,
        **kwargs,
    ) -> list[pm_types.PromptWrapper]:
        current_prompt = history[-1][-1]
        other_attempts = list(
            {
                html.escape(p.get_prompt_str()): (
                    p,
                    (p.extra.get("hypothesis") or "") if p.extra else "",
                )
                for ps in history
                for p in ps
                if p.get_prompt_str() != current_prompt.get_prompt_str()
            }.values()
        )[-5:]

        annotated_results = self._format_results(results)
        async with ls.trace("Optimize") as rt:
            print(f"Optimizing with url {rt.get_url()}", flush=True)
            formatted = current_prompt.get_prompt_str_in_context()
            hypo = (
                current_prompt.extra.get("hypothesis") if current_prompt.extra else None
            )
            if hypo:
                hypo = "Hypothesis for this prompt: " + hypo
            inputs = self.format(
                current_prompt=formatted,
                current_hypo=hypo or "",
                annotated_results=annotated_results,
                task_description=task.describe(),
                other_attempts=(
                    "\n\n---".join(
                        [
                            f"<hypothesis ix={i}>{hypo}</hypothesis>"
                            f"<attempt ix={i}>\n{p.get_prompt_str()}\n</attempt>"
                            for i, (p, hypo) in enumerate(other_attempts)
                        ]
                    )
                    if other_attempts
                    else "N/A"
                ),
            )
            prompt_output = await self.react_agent(inputs, current_prompt)
            rt.add_outputs({"output": prompt_output})
        candidate = pm_types.PromptWrapper.from_prior(
            current_prompt,
            prompt_output.improved_prompt,
            extra_info={"hypothesis": prompt_output.hypothesis},
        )

        pm_utils.print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt",
        )

        return [candidate]

    @ls.traceable
    async def react_agent(
        self, inputs: str, current_prompt, n=5
    ) -> pm_types.OptimizedPromptOutput:
        messages = [
            {"role": "user", "content": inputs},
        ]
        tooly = pm_types.prompt_schema(current_prompt)
        just_think = create_extractor(
            self.model,
            tools=[think, critique],
            tool_choice="any",
        )
        any_chain = create_extractor(
            self.model,
            tools=[think, critique, tooly],
            tool_choice="any",
        )
        final_chain = create_extractor(
            self.model,
            tools=[tooly],
            tool_choice="OptimizedPromptOutput",
        )
        for ix in range(n):
            if ix == n - 1:
                chain = final_chain
            elif ix == 0:
                chain = just_think
            else:
                chain = any_chain
            response = await chain.ainvoke(messages)
            final_response = next(
                (
                    r
                    for r in response["responses"]
                    if r.__repr_name__() == "OptimizedPromptOutput"
                ),
                None,
            )
            if final_response:
                return final_response
            msg: AIMessage = response["messages"][-1]
            messages.append(msg)
            ids = [tc["id"] for tc in (msg.tool_calls or [])]
            for id_ in ids:
                messages.append({"role": "tool", "content": "", "tool_call_id": id_})

        raise ValueError(f"Failed to generate response after {n} attempts")


def think(thought: str):
    """First call this to reason over complicated domains, uncover hidden input/output patterns, theorize why previous hypotheses failed, and creatively conduct error analyses (e.g., deep diagnostics/recursively analyzing "why" something failed). List characteristics of the data generating process you failed to notice before. Hypothesize fixes, prioritize, critique, and repeat calling this tool until you are confident in your next solution."""
    return "Take as much time as you need! If you're stuck, take a step back and try something new."


def critique(criticism: str):
    """Then, critique your thoughts and hypotheses. Identify flaws in your previous hypotheses and current thinking. Forecast why the hypotheses won't work. Get to the bottom of what is really driving the problem. This tool returns no new information but gives you more time to plan."""
    return "Take as much time as you need. It's important to think through different strategies."
