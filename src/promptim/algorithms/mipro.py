"""Mipro algo. Adapted from 

From: https://arxiv.org/abs/2406.11695
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from trustcall import create_extractor
from typing import (
    List,
    Dict,
    Optional,
    Union,
)
import uuid
import json

import langsmith as ls

from promptim.algorithms.base import BaseAlgorithm, AlgorithmConfig
from promptim.trainer import PromptTrainer
from promptim import types as pm_types
from promptim.algorithms.tpe_sampler import TPESampler


@dataclass
class MiproAlgorithmConfig(AlgorithmConfig):
    """
    Only the relevant parameters are included.
    No changes except renaming max_bootstrapped_examples -> max_bootstrapped_examples, etc.
    """

    max_bootstrapped_examples: int = 4
    max_labeled_examples: int = 4
    num_trials: int = 30
    minibatch: bool = True
    minibatch_size: int = 50
    minibatch_full_eval_steps: int = 10
    num_instruction_candidates: int = 10
    max_errors: int = 10
    requires_permission_to_run: bool = False
    seed: int = 42
    min_minibatch_size: int = 50


class MIPROAlgorithm(BaseAlgorithm[MiproAlgorithmConfig]):

    config_cls = MiproAlgorithmConfig

    @ls.traceable(name="MIPRO.run")
    async def run(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        initial_population: Union[pm_types.PromptWrapper, List[pm_types.PromptWrapper]],
        train_examples: List[pm_types.Example],
        dev_examples: List[pm_types.Example],
        *,
        system_config: Optional[dict] = None,
        annotation_queue: Optional[str] = None,
        commit_prompts: bool = False,
        experiment_name: str = "MIPRO Optimization",
        baseline_scores: Optional[dict] = None,
        baseline_experiment_results: Optional[list] = None,
    ) -> tuple[pm_types.PromptWrapper, float]:
        print("\n=== MIPRO: Starting Optimization ===")
        try:
            print(ls.get_current_run_tree().get_url())
        except Exception:
            pass

        cfg = self.config
        max_errors = cfg.max_errors

        # Handle initial_population
        if isinstance(initial_population, pm_types.PromptWrapper):
            initial_population = [initial_population]
        baseline_prompt = initial_population[-1]

        # Seed
        self._rng = random.Random(cfg.seed)

        # Evaluate baseline
        if baseline_scores and len(baseline_scores) > 0:
            best_score = sum(baseline_scores.values()) / len(baseline_scores)
        else:
            best_score = await self._evaluate(
                trainer, baseline_prompt, task, dev_examples, system_config
            )

        best_prompt = baseline_prompt

        example_candidates, max_errors = await self._synth_fewshots(
            trainer=trainer,
            task=task,
            train_examples=train_examples,
            rng=self._rng,
            cfg=cfg,
            teacher=best_prompt,
            system_config=system_config,
            max_errors=max_errors,
        )

        instructions = await self._propose_instructions(
            trainer=trainer,
            task=task,
            baseline_prompt=baseline_prompt,
            train_examples=train_examples,
            rng=self._rng,
            cfg=cfg,
            system_config=system_config,
        )
        instruction_candidates = [instructions]

        best_prompt = await self._optimize_prompt_parameters(
            trainer=trainer,
            task=task,
            dev_examples=dev_examples,
            baseline_prompt=baseline_prompt,
            example_candidates=example_candidates,
            instruction_candidates=instruction_candidates,
            best_score=best_score,
            system_config=system_config,
        )

        final_score = await self._evaluate(
            trainer, best_prompt, task, dev_examples, system_config
        )

        if commit_prompts:
            best_prompt.push_prompt(client=trainer.client)

        return best_prompt, final_score

    @ls.traceable
    async def _synth_fewshots(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        train_examples: List[pm_types.Example],
        rng: random.Random,
        cfg: MiproAlgorithmConfig,
        teacher: Optional[pm_types.PromptWrapper] = None,
        system_config: Optional[dict] = None,
        max_errors: int = 10,
    ) -> tuple[Optional[List[List[pm_types.Example]]], int]:
        max_boot = cfg.max_bootstrapped_examples
        max_label = cfg.max_labeled_examples
        if max_boot == 0 and max_label == 0:
            return None, max_errors

        if not train_examples:
            raise ValueError("Trainset cannot be empty.")
        num_sets = cfg.num_instruction_candidates

        all_candidates: List[List[pm_types.Example]] = []
        perm = train_examples.copy()
        rng.shuffle(perm)

        chunk_size = max_boot + max_label
        sets_created = 0
        i = 0
        while sets_created < num_sets and i < len(perm):
            subset = perm[i : i + chunk_size]
            i += chunk_size
            if not subset:
                break
            try:
                synth_examples = await self._build_fewshots(
                    trainer=trainer,
                    task=task,
                    teacher=teacher,
                    examples=subset,
                    max_boot=max_boot,
                    max_label=max_label,
                    system_config=system_config,
                )
                if synth_examples:
                    all_candidates.append(synth_examples)
                    sets_created += 1
            except Exception as e:
                # If teacher fails on some chunk, we skip it
                if max_errors <= 0:
                    raise e
                else:
                    max_errors -= 1

        if not all_candidates:
            return None, max_errors

        return all_candidates, max_errors

    @ls.traceable
    async def _build_fewshots(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        teacher: Optional[pm_types.PromptWrapper],
        examples: List[pm_types.Example],
        max_boot: int,
        max_label: int,
        system_config: Optional[dict] = None,
    ) -> List[pm_types.Example]:
        if teacher is None:
            return examples[:max_label]

        labeled_examples = []
        teacher_examples = []
        if max_label > 0 and len(examples) > 0:
            labeled_examples = examples[:max_label]

        teacher_inps = examples[max_label : max_label + max_boot]
        if teacher_inps:
            teacher_outputs = await self._call_teacher_for_synth_fewshots(
                trainer, task, teacher, teacher_inps, system_config
            )
            teacher_examples = teacher_outputs

        return labeled_examples + teacher_examples

    @ls.traceable
    async def _call_teacher_for_synth_fewshots(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        teacher: pm_types.PromptWrapper,
        examples: List[pm_types.Example],
        system_config: Optional[dict] = None,
    ) -> List[pm_types.Example]:
        """Calls teacher LLM on each example, capturing teacher's output as the reference."""
        results = await trainer._evaluate_prompt(
            teacher,
            task,
            examples,
            debug=self.config.debug,
            system_config=system_config,
        )
        teacher_examples = []
        for row in results:
            original_ex = row["example"]
            teacher_out = row["run"].outputs
            new_ex = pm_types.Example(
                id=uuid.uuid5(uuid.NAMESPACE_OID, str(original_ex.id) + "_teacher"),
                inputs=original_ex.inputs,
                outputs=teacher_out,
                metadata={"bootstrapped": True},
            )
            teacher_examples.append(new_ex)
        return teacher_examples

    @ls.traceable
    async def _propose_instructions(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        baseline_prompt: pm_types.PromptWrapper,
        train_examples: List[pm_types.Example],
        rng: random.Random,
        cfg: MiproAlgorithmConfig,
        system_config: Optional[dict] = None,
    ) -> List[str]:
        data_summary = await self._build_data_summary(
            trainer, task, train_examples, rng, cfg, system_config
        )

        instructions_prompt = pm_types.PromptWrapper.from_prior(
            baseline_prompt,
            (
                "You are an expert at analyzing training data to propose improved instructions. "
                "Below is a data summary and sample of training examples, along with a baseline prompt. "
                "Propose up to 10 new instructions (one per line) that will improve accuracy.\n\n"
                "<DATA_SUMMARY>\n"
                f"{data_summary}\n"
                "</DATA_SUMMARY>\n\n"
                "<TRAIN_DATA_SNIPPET>\n"
                f"{_sample_data_for_prompt(train_examples, max_len=5)}\n"
                "</TRAIN_DATA_SNIPPET>\n\n"
                "<BASELINE_PROMPT>\n"
                f"{baseline_prompt.get_prompt_str()}\n"
                "</BASELINE_PROMPT>\n\n"
                "Focus on both general patterns from the data summary and specific requirements "
                "from the examples. Each instruction should be clear and actionable.\n\n"
                "Now propose instructions:\n"
            ),
            extra_info={"proposer": True},
        )

        propose_results = await trainer._evaluate_prompt(
            instructions_prompt,
            task,
            train_examples,
            debug=cfg.debug,
            system_config=system_config,
            upload_results=False,
        )
        if not propose_results:
            return []

        raw_output = propose_results[0]["run"].outputs
        lines = [line for line in raw_output if line]
        instructions = {}
        for line in lines:
            instructions[json.dumps(line, sort_keys=True)] = line
            if len(instructions) >= cfg.num_instruction_candidates:
                break

        return list(instructions.values())

        # @ls.traceable
        # async def _propose_instructions(
        #     self,
        #     trainer: "PromptTrainer",
        #     task: pm_types.Task,
        #     baseline_prompt: pm_types.PromptWrapper,
        #     train_examples: List[pm_types.Example],
        #     rng: random.Random,
        #     cfg: "MiproAlgorithmConfig",
        #     system_config: Optional[dict] = None,
        #     max_summary_examples: int = 8,
        #     max_instructions: int = 10,
        # ) -> List[str]:
        #     """
        #     A robust method to propose new instructions for MIPRO:
        #     1) Summarize the training data, either with a multi-step chain or a single call.
        #     2) Form a meta-prompt that includes the data summary + the current baseline prompt.
        #     3) Directly call the LLM (no dummy example) to get a textual block of instructions.
        #     4) Parse them into a list. Return the instructions.
        #     """

        #     # 1) Summarize the training data (can be multi-step or a single call).
        #     data_summary = await self._build_data_summary(
        #         trainer,
        #         task,
        #         baseline_prompt,
        #         train_examples,
        #         max_summary_examples,
        #         system_config,
        #     )

        #     # 2) Build a meta-prompt that references the data summary and current prompt.
        #     #    We ask the LLM for up to `max_instructions` lines.
        #     meta_prompt_text = f"""\
        # You are an expert at analyzing training data and improving prompt instructions.

        # ## Data Summary
        # {data_summary}

        # ## Baseline Prompt
        # {baseline_prompt.get_prompt_str_in_context()}

        # Propose up to {max_instructions} new instructions (one per line) that will improve accuracy
        # and fix any weaknesses in the baseline prompt. Each instruction should be clear, actionable,
        # and mindful of the data summary. Do not repeat the baseline prompt's existing instructions.
        # """

        #     # 3) Now we do a direct call to the LLM.
        #     #    Instead of `_evaluate_prompt(...)`, we can do your own chain-of-thought or direct `_model_invoke`.
        #     #    If you prefer multi-step reasoning, see the chain-of-thought snippet below.
        #     #    For brevity, here's a single direct call.
        #     #    NOTE: You might define `self.model` or a custom approach.
        #     #    We'll show a direct "extractor" approach with no dummy examples.
        #     chain = create_extractor(
        #         self.model,  # or trainer.model
        #         tools=[],
        #         tool_choice="any",  # no tools, we just want the final LLM output
        #     )
        #     messages = [{"role": "user", "content": meta_prompt_text}]
        #     response = await chain.ainvoke(messages)

        #     # 4) We parse the raw textual output into instructions (line by line).
        #     #    Suppose the final LLM output is in `response["responses"][-1].text`.
        #     #    Or if you prefer:  next((r for r in response["responses"] if ...), None)
        #     if not response["responses"]:
        #         return []

        #     llm_output = response["responses"][
        #         -1
        #     ].text  # or .content if your library uses that
        #     lines = [l.strip() for l in llm_output.split("\n") if l.strip()]

        #     # Remove duplicates; keep up to `max_instructions`
        #     instructions_seen = {}
        #     final_instructions = []
        #     for line in lines:
        #         # a simple dedup
        #         if line not in instructions_seen:
        #             instructions_seen[line] = True
        #             final_instructions.append(line)
        #             if len(final_instructions) >= max_instructions:
        #                 break

        #     return final_instructions

    @ls.traceable
    async def _build_data_summary(
        self,
        baseline_prompt: pm_types.PromptWrapper,
        train_examples: List[pm_types.Example],
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    "Analyze these training examples to understand the data generating process. "
                    "Write detailed observations about trends that hold across most or all samples.\n\n"
                    "Consider:\n"
                    "- Topics and content patterns\n"
                    "- Syntax and structural patterns\n"
                    "- Conciseness and style patterns\n"
                    "- The likely nature/purpose of the task\n"
                    "- Any unstated assumptions or requirements\n\n"
                    "- Think how the prompt would know how to generate the exact correct output when restricted to the input variables.\n"
                    "Don't just list patterns - try to understand what they reveal about the underlying task. Solve all hidden patterns."
                    "Be creative in hypothesizing the data generation process.\n\n"
                    "Training examples:\n"
                    f"{_sample_data_for_prompt(train_examples, max_len=10)}\n\n"
                    f"Current prompt:\n{baseline_prompt.get_prompt_str()}"
                ),
            }
        ]

        def think(thought: str):
            """First call this to reason over complicated domains, uncover hidden input/output patterns, theorize why previous hypotheses failed, and creatively conduct error analyses (e.g., deep diagnostics/recursively analyzing "why" something failed). List characteristics of the data generating process you failed to notice before. Hypothesize fixes, prioritize, critique, and repeat calling this tool until you are confident in your next solution."""
            return "Take as much time as you need! If you're stuck, take a step back and try something new."

        def critique(criticism: str):
            """Then, critique your thoughts and hypotheses. Identify flaws in your previous hypotheses and current thinking. Forecast why the hypotheses won't work. Get to the bottom of what is really driving the problem. This tool returns no new information but gives you more time to plan."""
            return "Take as much time as you need. It's important to think through different strategies."

        def summarize(summary: str) -> str:
            """Create a final summary of the analysis that captures key patterns and insights about the data."""
            return ""

        data_summary = ""
        max_steps = 5

        for step in range(max_steps):
            if step == max_steps - 1:
                chain = create_extractor(
                    self.model,
                    tools=[summarize],
                    tool_choice="summarize",
                )
            elif step == 0:
                chain = create_extractor(
                    self.model,
                    tools=[think, critique],
                    tool_choice="any",
                )
            else:
                chain = create_extractor(
                    self.model,
                    tools=[think, critique, summarize],
                    tool_choice="any",
                )

            response = await chain.ainvoke(messages)

            final_response = next(
                (r for r in response["responses"] if r.__repr_name__() == "summarize"),
                None,
            )
            if final_response:
                data_summary = final_response
                break

            msg = response["messages"][-1]
            messages.append(msg)

            ids = [tc["id"] for tc in (msg.tool_calls or [])]
            for id_ in ids:
                messages.append({"role": "tool", "content": "", "tool_call_id": id_})
        return data_summary

    @ls.traceable
    async def _optimize_prompt_parameters(
        self,
        trainer: PromptTrainer,
        task: pm_types.Task,
        dev_examples: List[pm_types.Example],
        baseline_prompt: pm_types.PromptWrapper,
        example_candidates: Optional[List[List[pm_types.Example]]],
        instruction_candidates: Dict[int, List[str]],
        best_score: float,
        system_config: Optional[dict] = None,
    ) -> pm_types.PromptWrapper:
        """ "Pick an instruction index and a few-shot index, merge them into a candidate prompt,
        and do partial or full dev eval, and record the score.
        """
        cfg: MiproAlgorithmConfig = self.config
        best_prompt: pm_types.PromptWrapper = baseline_prompt

        self._score_data = []
        self._score_data.append((best_score, baseline_prompt, True))
        self._best_score = best_score

        dev_subset = dev_examples
        if cfg.minibatch and len(dev_examples) > cfg.min_minibatch_size:
            dev_subset = self._rng.sample(dev_examples, cfg.minibatch_size)

        async def objective(trial: TPESampler) -> float:
            nonlocal best_prompt

            i = 0
            instr_list = instruction_candidates[i]
            instr_idx = trial.suggest_int(
                f"inst_idx_{i}",
                0,
                len(instr_list) - 1,
                n_candidates=24,
                gamma=0.2,
                lower_is_better=False,
            )
            chosen_instruction = instr_list[instr_idx]

            # For synth_examples
            demos_idx = None
            if example_candidates:
                demos_idx = trial.suggest_int(
                    "demo_idx",
                    0,
                    len(example_candidates) - 1,
                    n_candidates=24,
                    gamma=0.2,
                    lower_is_better=False,
                )
                chosen_demo = example_candidates[demos_idx]
            else:
                chosen_demo = None

            candidate_prompt = _merge_instruction_and_examples(
                baseline_prompt,
                chosen_instruction,
                chosen_demo,
            )

            # Evaluate partial or full
            if cfg.minibatch:
                score = await self._evaluate(
                    trainer, candidate_prompt, task, dev_subset, system_config
                )
            else:
                score = await self._evaluate(
                    trainer, candidate_prompt, task, dev_examples, system_config
                )

            trial.register(f"inst_idx_{i}", float(instr_idx), float(score))
            if example_candidates:
                trial.register("demo_idx", float(demos_idx), float(score))

            self._score_data.append((score, candidate_prompt, not cfg.minibatch))

            if cfg.minibatch:
                trial_num = len(trial.observations.get("inst_idx_0", []))
                if (trial_num % cfg.minibatch_full_eval_steps == 0) or (
                    trial_num == cfg.num_trials
                ):
                    # pick best partial so far
                    best_partial = max(self._score_data, key=lambda x: x[0])
                    # do full eval on that prompt if not already
                    if not best_partial[2]:
                        best_partial_score = asyncio.run(
                            self._evaluate(
                                trainer,
                                best_partial[1],
                                task,
                                dev_examples,
                                system_config,
                            )
                        )
                        self._score_data.append(
                            (best_partial_score, best_partial[1], True)
                        )
                        if best_partial_score > self._best_score:
                            self._best_score = best_partial_score
                            best_prompt = best_partial[1]

            else:
                if score > self._best_score:
                    self._best_score = score
                    best_prompt = candidate_prompt

            return score

        sampler = TPESampler(seed=cfg.seed)
        _ = await sampler.optimize(objective, n_trials=cfg.num_trials)

        fully_evaled = [x for x in self._score_data if x[2]]
        if not fully_evaled:
            candidate = max(self._score_data, key=lambda x: x[0])
            best_prompt = candidate[1]
            self._best_score = candidate[0]
        else:
            candidate = max(fully_evaled, key=lambda x: x[0])
            best_prompt = candidate[1]
            self._best_score = candidate[0]

        best_prompt.extra = best_prompt.extra or {}
        best_prompt.extra["mipro_logs"] = {
            "score_data": [(float(s), str(p), fe) for (s, p, fe) in self._score_data],
            "best_score": float(self._best_score),
        }

        return best_prompt

    async def _evaluate(
        self,
        trainer: PromptTrainer,
        prompt: pm_types.PromptWrapper,
        task: pm_types.Task,
        dev_examples: List[pm_types.Example],
        system_config: Optional[dict] = None,
    ) -> float:
        results = await trainer._evaluate_prompt(
            prompt,
            task,
            data=dev_examples,
            debug=self.config.debug,
            system_config=system_config,
        )
        scores = await trainer.calculate_scores(results)
        if not scores:
            return float("-inf")
        return sum(scores.values()) / len(scores)


def _sample_data_for_prompt(examples: List[pm_types.Example], max_len: int = 5) -> str:
    subset = examples[:max_len]
    lines = []
    for ex in subset:
        lines.append(f"Inputs={ex.inputs}, Reference={ex.outputs}")
    return "\n".join(lines)


def _merge_instruction_and_examples(
    baseline_prompt: pm_types.PromptWrapper,
    instruction: str,
    synth_examples: Optional[List[pm_types.Example]],
) -> pm_types.PromptWrapper:
    old_text = baseline_prompt.get_prompt_str_in_context()
    new_text = f"{old_text}\n\n# Additional Instruction:\n{instruction}"

    def sanitize(s):
        return str(s).replace("{", "{{").replace("}", "}}")

    if synth_examples:
        new_text += "\n\n# Few-Shot Examples:\n"
        for d in synth_examples:
            outputs = d.outputs
            if (
                isinstance(outputs, dict)
                and len(outputs) == 1
                and next(iter(outputs)) == "output"
            ):
                outputs = outputs["output"]
            new_text += (
                f"Input: {sanitize(d.inputs)}\nIdeal Output: {sanitize(outputs)}\n---\n"
            )

    candidate_prompt = pm_types.PromptWrapper.from_prior(
        baseline_prompt,
        new_text,
        extra_info={
            "instruction_used": instruction,
            "fewshot_count": len(synth_examples) if synth_examples else 0,
        },
    )
    return candidate_prompt
