from typing import Optional, Literal, Sequence, cast
from langsmith.evaluation._arunner import ExperimentResultRow
from dataclasses import dataclass, field
from promptim import types as pm_types, _utils as pm_utils
from promptim.optimizers import base as optimizers
from pydantic import BaseModel, Field
import langsmith as ls
import random
from promptim.optimizers.metaprompt import DEFAULT_METAPROMPT
from trustcall import create_extractor

_DEFAULT_RECOMMENDATION_PROMPT = """You are an expert AI assistant. Your task is to analyze a failing test case for a given prompt and provide specific recommendations for how to improve the prompt.

The current prompt is:
<current_prompt>
{prompt}
</current_prompt>

You will be provided with details of a failing example.
Analyze the test case, the current prompt, and any evaluation scores.
Develop a theory of why the model failed for this example. Perform a detailed analysis.
Then, provide targeted recommendations for improvements to the prompt.

YOU MUST RESPOND WITH A JSON OBJECT THAT VALIDATES ACCORDING TO THE 'Advise' TOOL SCHEMA.
The JSON object must have two keys: 'analysis' (a string analyzing the failure) and 'recommended_changes' (a string with specific recommendations for the prompt).

Example of a VAlID JSON response:
{{
  "analysis": "The prompt failed because it was too vague about X, leading the model to misunderstand Y.",
  "recommended_changes": "Modify the prompt to explicitly state Z. Add an example for condition W."
}}

Do not include any other text or explanation outside of this JSON object.
"""


@dataclass(kw_only=True)
class Config(optimizers.Config):
    kind: Literal["feedback_guided"] = field(
        default="feedback_guided",
        metadata={
            "description": "The feedback_guided optimizer  that iteratively improves"
            " prompts based on feedback from evaluation results, focusing on examples that fall below a specified performance threshold."
        },
    )
    recommendation_prompt: str = field(
        default=_DEFAULT_RECOMMENDATION_PROMPT,
    )
    score_threshold: float = 0.8
    max_batch_size: Optional[int] = 20


class Advise(BaseModel):
    """Think step-by-step, analyzing the task and test results. Provide a clear recommendation on why the prompt failed this
    test case, and what it should do to succeed next time for this type of input. Focus on the test metrics and expected output (if available).
    """

    analysis: str = Field(
        description="First, analyze why the prompt failed for this example. Think of what instructions in the prompt were poorly defined or missing."
    )
    recommended_changes: str = Field(
        description="Second, provide targeted recommendations for improvements."
    )


class FeedbackGuidedOptimizer(optimizers.BaseOptimizer):
    """
    A two-phase optimization algorithm that:
    1. Identifies examples with scores below a threshold
    2. Generates targeted recommendations for improvements
    3. Uses these recommendations to guide prompt refinement

    The algorithm is particularly effective when you want to focus
    optimization efforts on specific failure cases while maintaining
    overall prompt quality.
    """

    config_cls = Config

    def __init__(
        self,
        *,
        model: optimizers.MODEL_TYPE | None = None,
        score_threshold: float = 0.8,
        recommendation_prompt: Optional[str] = None,
        meta_prompt: Optional[str] = None,
        max_batch_size: Optional[int] = 20,
    ):
        super().__init__(model=model)
        self.score_threshold = score_threshold
        self.recommendation_prompt = (
            recommendation_prompt or _DEFAULT_RECOMMENDATION_PROMPT
        )
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT
        self.max_batch_size = max_batch_size

    def _format_failing_examples(
        self, results: list[ExperimentResultRow]
    ) -> list[dict]:
        """Identify and format examples that fall below the score threshold."""
        failing = []
        for r in results:
            # Consider "failing" if any evaluation score is below threshold
            if any(
                (
                    eval_result.score is not None
                    and eval_result.score < self.score_threshold
                )
                for eval_result in r["evaluation_results"]["results"]
            ):
                failing.append(self._format_example(r))
        return failing

    def _format_example(self, example: ExperimentResultRow) -> str:
        """Format failing examples into a string for analysis."""
        outputs = example["example"].outputs

        if outputs:
            ref_output = f"But we expected: {outputs}"
        else:
            ref_output = ""
        scores = []
        for eval_result in example["evaluation_results"]["results"]:
            scores.append(
                f"- {eval_result.key}: {eval_result.score}"
                f"{f' (Comment: {eval_result.comment})' if eval_result.comment else ''}"
            )

        scores = "\n".join(scores)
        if scores:
            scores = f"\n\nTest results:\n{scores}"

        return f"""Failing Example:
For input: {example['example'].inputs}
The prompt predicted: {example['run'].outputs}
{ref_output}
{scores}
"""

    async def improve_prompt(
        self,
        history: Sequence[Sequence[pm_types.PromptWrapper]],
        results: list[ExperimentResultRow],
        task: pm_types.Task,
        **kwargs,
    ) -> list[pm_types.PromptWrapper]:
        """Improve prompt using feedback from failing examples.

        1. Select failing examples
        2. If no failing examples, return current prompt
        3. Batch advisor over failing examples
        4. Format advisor responses into a string
        5. Run metaprompt over formatted advice
        """
        current_prompt = history[-1][-1]
        other_attempts = [
            p for prompts in history for p in prompts if p is not current_prompt
        ]
        # 1. Identify failing examples
        failing_examples = self._format_failing_examples(results)

        # 2. If no failing examples, return current prompt unchanged
        if not failing_examples:
            return list(history[-1])
        if self.max_batch_size and len(failing_examples) > self.max_batch_size:
            random.shuffle(failing_examples)
            failing_examples = failing_examples[: self.max_batch_size]
        # 3. Generate targeted recommendations for each failing example
        advisor = None # Initialize advisor
        try:
            advisor = create_extractor(self.model, tools=[Advise])
            print(f"DEBUG: Successfully created advisor object: {type(advisor)}")
        except Exception as e_create_extractor:
            print(f"ERROR: Exception during create_extractor for advisor: {type(e_create_extractor).__name__}: {e_create_extractor}")
            import traceback
            print(traceback.format_exc())
            return list(history[-1]) # Cannot proceed without advisor
        
        if not advisor: # Should be caught by the except block, but as a safeguard
            print("ERROR: Advisor object is None after create_extractor, cannot proceed.")
            return list(history[-1])

        advisor_inputs = [] # Initialize
        try:
            # Moved list comprehension inside try block
            advisor_inputs = [
                [
                    (
                        "system",
                        self.recommendation_prompt.format(
                            prompt=current_prompt.get_prompt_str_in_context()
                        ),
                    ),
                    ("user", example),
                ]
                for example in failing_examples
            ]
            print(f"DEBUG: Successfully created advisor_inputs. Number of inputs: {len(advisor_inputs)}")
            if advisor_inputs: print(f"DEBUG: First advisor_input (system part snippet): {str(advisor_inputs[0][0])[:200]}...")

            # The ls.trace block and abatch call are now also in this try-except
            with ls.trace(
                name="Analyze examples", inputs={"num_examples": len(failing_examples)}
            ):
                results_ = None # Initialize results_
                # try-except for abatch is still here, nested
                try:
                    results_ = await advisor.abatch(advisor_inputs)
                except Exception as e_abatch:
                    print(f"ERROR: Exception during advisor.abatch(advisor_inputs): {type(e_abatch).__name__}: {e_abatch}")
                    import traceback
                    print(traceback.format_exc()) # Print full traceback
                    # results_ will remain None
    
                print(f"DEBUG: Raw results_ from advisor.abatch: {type(results_)}")
                if isinstance(results_, list) and results_:
                    print(f"DEBUG: First item of results_: {str(results_[0])[:500]}...")
                elif results_:
                     print(f"DEBUG: results_ (not a list or empty): {str(results_)[:500]}...")
                else:
                    print(f"DEBUG: results_ is empty or None")
    
                recommendations = []
                if results_:
                    for r_idx, r_val in enumerate(results_):
                        print(f"DEBUG: Processing r_val item {r_idx}: {str(r_val)[:500]}...")
                        parsed_advise = None
                        # Attempt 1: Check 'responses' field (current logic)
                        if isinstance(r_val, dict) and r_val.get("responses") and isinstance(r_val["responses"], list) and len(r_val["responses"]) > 0:
                            response_item = r_val["responses"][0]
                            try:
                                if hasattr(response_item, 'model_dump') and callable(getattr(response_item, 'model_dump')): # Pydantic model
                                    parsed_advise = Advise(**response_item.model_dump())
                                elif isinstance(response_item, dict):
                                    parsed_advise = Advise(**response_item)
                            except Exception as e_advise_responses: # More specific exception name
                                print(f"Warning: Could not create/cast to Advise from 'responses' for item {r_idx}: {type(e_advise_responses).__name__}: {e_advise_responses}. Item: {response_item}")
                        
                        # Attempt 2: If 'responses' is empty or failed, check 'messages' for 'tool_calls'
                        if not parsed_advise and isinstance(r_val, dict) and r_val.get("messages") and isinstance(r_val["messages"], list) and len(r_val["messages"]) > 0:
                            first_message = r_val["messages"][0]
                            if hasattr(first_message, 'tool_calls') and first_message.tool_calls and isinstance(first_message.tool_calls, list) and len(first_message.tool_calls) > 0:
                                tool_call = first_message.tool_calls[0]
                                if hasattr(tool_call, 'args'):
                                    try:
                                        if isinstance(tool_call.args, dict):
                                            print(f"DEBUG: Advisor tool_call.args (dict): {tool_call.args}") # ADDED
                                            parsed_advise = Advise(**tool_call.args)
                                        elif isinstance(tool_call.args, str):
                                            import json
                                            try:
                                                args_dict = json.loads(tool_call.args)
                                                print(f"DEBUG: Advisor tool_call.args (string) successfully parsed to dict: {args_dict}")
                                                parsed_advise = Advise(**args_dict)
                                            except json.JSONDecodeError as e_json:
                                                print(f"Warning: tool_call.args for item {r_idx} is a string but not valid JSON: {tool_call.args}. Error: {e_json}")
                                    except Exception as e_tool_call_advise: # More specific exception name
                                        print(f"Warning: Could not create Advise from tool_call.args for item {r_idx}: {type(e_tool_call_advise).__name__}: {e_tool_call_advise}. Args: {tool_call.args}")
                                else:
                                    print(f"Warning: tool_call for item {r_idx} does not have 'args'. Tool_call: {tool_call}")
                            elif hasattr(first_message, 'content') and isinstance(first_message.content, str):
                                try:
                                    import json
                                    content_dict = json.loads(first_message.content)
                                    print(f"DEBUG: Advisor first_message.content successfully parsed to dict: {content_dict}") # ADDED
                                    if isinstance(content_dict, dict) and "analysis" in content_dict and "recommended_changes" in content_dict:
                                       parsed_advise = Advise(**content_dict)
                                except json.JSONDecodeError:
                                    # Content is not JSON or not matching Advise structure, ignore for auto-parsing
                                    # print(f"DEBUG: Advisor first_message.content is not a valid JSON or Advise structure: {first_message.content!r}") # Optional: very verbose
                                    pass 
                                except Exception as e_content_parse:
                                    print(f"Warning: Tried parsing message content as Advise for item {r_idx} but failed: {type(e_content_parse).__name__}: {e_content_parse}")
    
                        if parsed_advise:
                            recommendations.append(parsed_advise)
                            print(f"DEBUG: Successfully parsed Advise object {r_idx}:")
                            if hasattr(parsed_advise, 'analysis'):
                                print(f"  DEBUG: Advise.analysis: {parsed_advise.analysis!r}")
                            else:
                                print(f"  DEBUG: Advise object has no 'analysis' attribute.")
                            if hasattr(parsed_advise, 'recommended_changes'):
                                print(f"  DEBUG: Advise.recommended_changes: {parsed_advise.recommended_changes!r}")
                            else:
                                print(f"  DEBUG: Advise object has no 'recommended_changes' attribute.")
                        else:
                            print(f"Warning: No valid Advise object could be parsed for recommendation item {r_idx}. Raw item (snippet): {str(r_val)[:200]}...") # ADDED snippet
                
                if not recommendations:
                    print("Warning: No valid recommendations were generated by the advisor LLM. Returning current prompt.")
                    return list(history[-1])
    
            formatted_recommendations = []
            for i, (example, rec) in enumerate(zip(failing_examples, recommendations)):
                # Ensure rec is an Advise instance and has the expected attributes
                if not isinstance(rec, Advise) or not hasattr(rec, 'recommended_changes'):
                    print(f"ERROR: Item {i} in recommendations is not a valid Advise object or lacks 'recommended_changes'. Item: {rec!r}. Skipping.")
                    continue
                formatted_recommendations.append(f"Recommended changes for example {i+1}:") # Corrected variable name
                formatted_recommendations.append(rec.recommended_changes)
                formatted_recommendations.append("-" * 40 + "\n")
    
            all_recommendations = "\n".join(formatted_recommendations)
            print(f"DEBUG: Consolidated all_recommendations for metaprompt: {all_recommendations!r}")
    
            chain = create_extractor(
                self.model,
                tools=[pm_types.prompt_schema(current_prompt)],
                tool_choice="OptimizedPromptOutput",
            )
            inputs = {
                "current_hypo": "",
                "current_prompt": current_prompt.get_prompt_str_in_context(),
                "task_description": task.describe(),
                "annotated_results": all_recommendations,
                "other_attempts": (
                    "\n\n---".join([p.get_prompt_str() for p in other_attempts])
                    if other_attempts
                    else "N/A"
                ),
            }
            with ls.trace("Apply Recommendations", inputs=inputs) as rt:
                prompt_output_raw = await chain.ainvoke(self.meta_prompt.format(**inputs))
                
                final_prompt_data = None
                # Ensure prompt_output_raw and prompt_output_raw["responses"] exist and are not empty
                if isinstance(prompt_output_raw, dict) and prompt_output_raw.get("responses") and isinstance(prompt_output_raw["responses"], list) and len(prompt_output_raw["responses"]) > 0:
                    final_prompt_data_item = prompt_output_raw["responses"][0]
                    try:
                        # Check if final_prompt_data_item is a Pydantic model that has the improved_prompt attribute
                        if hasattr(final_prompt_data_item, 'model_dump') and callable(getattr(final_prompt_data_item, 'model_dump')) and hasattr(final_prompt_data_item, 'improved_prompt') and final_prompt_data_item.improved_prompt:
                            print(f"DEBUG: final_prompt_data_item from 'responses' is a Pydantic-like model with a valid 'improved_prompt': {type(final_prompt_data_item)}")
                            # It's already the Pydantic model we need (or conforms to the protocol)
                            # No need to dump and reload; use it directly.
                            final_prompt_data = final_prompt_data_item
                            print(f"DEBUG: Successfully assigned final_prompt_data_item directly. Improved prompt snippet: {str(final_prompt_data.improved_prompt)[:200]}...")
                        elif isinstance(final_prompt_data_item, dict):
                            print(f"DEBUG: final_prompt_data_item from 'responses' is a dict.")
                            if final_prompt_data_item.get('improved_prompt'): # Must have improved_prompt
                                # Try to see if it matches all fields first for a more complete object
                                if all(k in final_prompt_data_item for k in pm_types.OptimizedPromptOutput.model_fields.keys()):
                                    print(f"DEBUG: Dict item matches all OptimizedPromptOutput fields. Creating from dict: { {k: str(v)[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in final_prompt_data_item.items()} }")
                                    try:
                                        final_prompt_data = pm_types.OptimizedPromptOutput(**final_prompt_data_item)
                                    except TypeError as e_protocol_dict:
                                        print(f"ERROR: TypeError when creating OptimizedPromptOutput from dict (likely protocol issue): {e_protocol_dict}. Falling back to direct use if structure matches.")
                                        # Fallback: if it was a dict matching the protocol's fields, try to use it as a structured dict (though less ideal)
                                        # This path is less likely to be hit if the above Pydantic check is robust.
                                        if 'improved_prompt' in final_prompt_data_item: # Basic check
                                            print("INFO: Using dict directly as it matches OptimizedPromptOutput structure (protocol instantiation failed).")
                                            # This isn't ideal as it won't be a Pydantic model, but might allow progress if structure is exact.
                                            # Consider this a temporary workaround; ideally, chain.ainvoke returns a usable Pydantic model.
                                            final_prompt_data = final_prompt_data_item # Using the dict directly
                                        else:
                                            final_prompt_data = None
                                    except Exception as e_dict_create:
                                        print(f"ERROR: Failed to create OptimizedPromptOutput from dict: {type(e_dict_create).__name__}: {e_dict_create}. Dict: {final_prompt_data_item}")
                                        final_prompt_data = None
                                else: # Fallback to partial creation if only improved_prompt is there
                                    print(f"DEBUG: Dict item has 'improved_prompt', attempting to create OptimizedPromptOutput partially: { {k: str(v)[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in final_prompt_data_item.items()} }")
                                    try:
                                        final_prompt_data = pm_types.OptimizedPromptOutput(
                                            improved_prompt=final_prompt_data_item['improved_prompt'],
                                            analysis=final_prompt_data_item.get('analysis'),
                                            hypothesis=final_prompt_data_item.get('hypothesis')
                                        )
                                    except TypeError as e_protocol_partial_dict:
                                        print(f"ERROR: TypeError when creating OptimizedPromptOutput partially from dict (likely protocol issue): {e_protocol_partial_dict}.")
                                        final_prompt_data = None
                                    except Exception as e_dict_partial_create:
                                        print(f"ERROR: Failed to create OptimizedPromptOutput partially from dict: {type(e_dict_partial_create).__name__}: {e_dict_partial_create}")
                                        final_prompt_data = None
                            else:
                                 print(f"Warning: final_prompt_data_item (dict) from 'responses' is missing 'improved_prompt' or it's empty. Item: {final_prompt_data_item}")
                                 final_prompt_data = None
                        else:
                            print(f"Warning: Unexpected final prompt data format in 'responses'. Type: {type(final_prompt_data_item)}. Item (snippet): {str(final_prompt_data_item)[:200]}...")
                            final_prompt_data = None
                    except Exception as e_op_output:
                        print(f"Warning: Could not create/cast to OptimizedPromptOutput from 'responses' pathway: {type(e_op_output).__name__}: {e_op_output}. Item (snippet): {str(final_prompt_data_item)[:200]}...")
                        final_prompt_data = None # Ensure final_prompt_data is None if any error occurs in this block
                # Attempt 2: If 'responses' is empty or failed, check 'messages' for 'tool_calls' for OptimizedPromptOutput
                elif isinstance(prompt_output_raw, dict) and prompt_output_raw.get("messages") and isinstance(prompt_output_raw["messages"], list) and len(prompt_output_raw["messages"]) > 0:
                    print("DEBUG: Trying 'messages' -> 'tool_calls' pathway for OptimizedPromptOutput.")
                    first_message = prompt_output_raw["messages"][0]
                    if hasattr(first_message, 'tool_calls') and first_message.tool_calls and isinstance(first_message.tool_calls, list) and len(first_message.tool_calls) > 0:
                        tool_call = first_message.tool_calls[0]
                        if hasattr(tool_call, 'args'):
                            try:
                                if isinstance(tool_call.args, dict):
                                    final_prompt_data = pm_types.OptimizedPromptOutput(**tool_call.args)
                                elif isinstance(tool_call.args, str):
                                    import json
                                    try:
                                        args_dict = json.loads(tool_call.args)
                                        if args_dict.get('improved_prompt'): # Must have improved_prompt
                                            final_prompt_data = pm_types.OptimizedPromptOutput(**args_dict)
                                        else:
                                            print(f"Warning: Metaprompt tool_call.args (JSON string) is missing 'improved_prompt'. Args: {args_dict}")
                                            final_prompt_data = None
                                    except json.JSONDecodeError:
                                         print(f"Warning: Metaprompt tool_call.args is a string but not valid JSON: {tool_call.args}")
                                         final_prompt_data = None
                            except Exception as e_meta_tool_call:
                                print(f"Warning: Could not create OptimizedPromptOutput from metaprompt tool_call.args: {e_meta_tool_call}. Args: {tool_call.args}")
                                final_prompt_data = None
                # Fallback: if the raw output itself is the dict for OptimizedPromptOutput (as seen in the log)
                elif isinstance(prompt_output_raw, dict) and prompt_output_raw.get('improved_prompt'): # Must have improved_prompt
                     print("DEBUG: Trying direct dict pathway for OptimizedPromptOutput.")
                     try:
                        final_prompt_data = pm_types.OptimizedPromptOutput(**prompt_output_raw)
                     except Exception as e_direct_parse:
                        print(f"Warning: Could not create OptimizedPromptOutput directly from prompt_output_raw: {type(e_direct_parse).__name__}: {e_direct_parse}. Item: {prompt_output_raw}")
                        final_prompt_data = None


                if not final_prompt_data or not hasattr(final_prompt_data, 'improved_prompt') or not final_prompt_data.improved_prompt: # Check improved_prompt is non-empty
                    print("Warning: Metaprompt did not return a valid improved prompt structure or improved_prompt field. Returning current prompt.")
                    # Log the raw output for debugging if it failed to parse
                    print(f"DEBUG: Raw prompt_output_raw from metaprompt (snippet): {str(prompt_output_raw)[:1000]}...")
                    return list(history[-1]) # Fallback if metaprompt output is invalid

                rt.add_outputs({"prompt_output": final_prompt_data})

        except Exception as e_outer:
            print(f"ERROR: Outer exception in improve_prompt before/during advisor data processing: {type(e_outer).__name__}: {e_outer}")
            import traceback
            print(traceback.format_exc())
            return list(history[-1]) # Fallback if there's an error in this broader block
        
        # This part is reached only if the above try-except for advisor_inputs and abatch completed without re-raising
        # and without returning early due to other caught exceptions.
        # The original metaprompt logic starts here.

        candidate = pm_types.PromptWrapper.from_prior(
            current_prompt, final_prompt_data.improved_prompt
        )

        pm_utils.print_rich_diff(
            current_prompt.get_prompt_str_in_context(),
            candidate.get_prompt_str_in_context(),
            "Updated Prompt with Targeted Improvements",
        )
        return [candidate]
