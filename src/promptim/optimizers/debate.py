from langsmith.evaluation._arunner import ExperimentResultRow
from promptim import types as pm_types
import json
from promptim.optimizers import base as optimizers
from dataclasses import dataclass


@dataclass
class Vulnerability:
    description: str
    severity: float  # 0-1 scale
    likelihood: float  # 0-1 scale
    examples: list[str]


@dataclass
class Defense:
    prompt_changes: str
    addressed_vulnerabilities: list[str]
    tradeoffs: list[str]
    confidence: float  # 0-1 scale


@dataclass
class DebateRound:
    consensus_points: list[str]
    disagreements: list[str]
    proposed_resolutions: list[str]
    confidence_ratings: dict[str, float]


class DebaterOptimizer(optimizers.BaseOptimizer):
    def __init__(
        self, model, consensus_threshold: float = 0.8, max_debate_rounds: int = 3
    ):
        super().__init__(model=model)
        self.consensus_threshold = consensus_threshold
        self.max_debate_rounds = max_debate_rounds

    def _format_vulnerabilities(self, vulnerabilities: list[Vulnerability]) -> str:
        formatted = []
        for i, vuln in enumerate(vulnerabilities, 1):
            formatted.append(
                f"""
Vulnerability {i}:
Description: {vuln.description}
Severity: {vuln.severity}
Likelihood: {vuln.likelihood}
Examples: {', '.join(vuln.examples)}"""
            )
        return "\n".join(formatted)

    def _format_defenses(self, defenses: list[Defense]) -> str:
        formatted = []
        for i, defense in enumerate(defenses, 1):
            formatted.append(
                f"""
Defense {i}:
Changes: {defense.prompt_changes}
Addresses: {', '.join(defense.addressed_vulnerabilities)}
Tradeoffs: {', '.join(defense.tradeoffs)}
Confidence: {defense.confidence}"""
            )
        return "\n".join(formatted)

    def _parse_vulnerabilities(self, response: str) -> list[Vulnerability]:
        try:
            json_str = response[response.find("{") : response.rfind("}") + 1]
            data = json.loads(json_str)
            vulnerabilities = []
            items = data.get("vulnerabilities", []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                items = [items]
            for item in items:
                vuln = Vulnerability(
                    description=str(item.get("description", "")),
                    severity=float(item.get("severity", 0.5)),
                    likelihood=float(item.get("likelihood", 0.5)),
                    examples=list(map(str, item.get("examples", []))),
                )
                vulnerabilities.append(vuln)
            return vulnerabilities
        except Exception as e:
            print(f"Error parsing vulnerabilities: {e}")
            return []

    def _parse_defenses(self, response: str) -> list[Defense]:
        try:
            json_str = response[response.find("[") : response.rfind("]") + 1]
            data = json.loads(json_str)
            defenses = []
            items = data if isinstance(data, list) else [data]
            for item in items:
                try:
                    defense = Defense(
                        prompt_changes=str(item.get("prompt_changes", "")),
                        addressed_vulnerabilities=list(
                            map(str, item.get("addressed_vulnerabilities", []))
                        ),
                        tradeoffs=list(map(str, item.get("tradeoffs", []))),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                    defenses.append(defense)
                except Exception as e:
                    print(f"Error parsing defense item: {e}")
                    continue
            return defenses
        except Exception as e:
            print(f"Error parsing defenses: {e}")
            print(f"Response was: {response}")  # debug print
            return []

    def _parse_debate_response(self, response: str) -> dict:
        try:
            if response.strip().startswith("["):
                json_str = response[response.find("[") : response.rfind("]") + 1]
                data = json.loads(json_str)[0]
            else:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                data = json.loads(json_str)
            return {
                "consensus_points": list(map(str, data.get("consensus_points", []))),
                "disagreements": list(map(str, data.get("disagreements", []))),
                "proposed_resolutions": list(
                    map(str, data.get("proposed_resolutions", []))
                ),
                "confidence_ratings": {
                    str(k): float(v)
                    for k, v in data.get("confidence_ratings", {}).items()
                },
            }
        except Exception as e:
            print(f"Error parsing debate response: {e}")
            print(f"Response was: {response}")  # debug print
            return {
                "consensus_points": [],
                "disagreements": [],
                "proposed_resolutions": [],
                "confidence_ratings": {},
            }

    def _synthesize_debate_responses(self, responses: list[dict]) -> DebateRound:
        all_consensus = set()
        all_disagreements = set()
        all_resolutions = set()
        confidence_sums = {}
        confidence_counts = {}
        for response in responses:
            all_consensus.update(response["consensus_points"])
            all_disagreements.update(response["disagreements"])
            all_resolutions.update(response["proposed_resolutions"])
            for resolution, confidence in response["confidence_ratings"].items():
                confidence_sums[resolution] = (
                    confidence_sums.get(resolution, 0) + confidence
                )
                confidence_counts[resolution] = confidence_counts.get(resolution, 0) + 1
        avg_confidence = {
            resolution: confidence_sums[resolution] / confidence_counts[resolution]
            for resolution in confidence_sums
        }
        return DebateRound(
            consensus_points=list(all_consensus),
            disagreements=list(all_disagreements),
            proposed_resolutions=list(all_resolutions),
            confidence_ratings=avg_confidence,
        )

    def _parse_test_results(self, response: str) -> Tuple[float, list[str]]:
        try:
            json_str = response[response.find("{") : response.rfind("}") + 1]
            data = json.loads(json_str)
            robustness_score = float(data.get("robustness_score", 0.5))
            weaknesses = list(map(str, data.get("identified_weaknesses", [])))
            return robustness_score, weaknesses
        except Exception as e:
            print(f"Error parsing test results: {e}")
            return 0.5, []

    def _apply_consensus_changes(
        self, prompt: pm_types.PromptWrapper, debate_result: DebateRound
    ) -> pm_types.PromptWrapper:
        high_confidence_resolutions = [
            res
            for res in debate_result.proposed_resolutions
            if debate_result.confidence_ratings.get(res, 0) > self.consensus_threshold
        ]
        if not high_confidence_resolutions:
            return prompt
        prompt_str = prompt.get_prompt_str()
        for resolution in high_confidence_resolutions:
            prompt_str = self._apply_resolution(prompt_str, resolution)
        return pm_types.PromptWrapper.from_prior(prompt, prompt_str)

    async def improve_prompt(
        self,
        current_prompt: pm_types.PromptWrapper,
        results: list[ExperimentResultRow],
        task: pm_types.Task,
        other_attempts: list[pm_types.PromptWrapper],
        trainer: "Trainer" = None,
    ) -> pm_types.PromptWrapper:
        print("\n=== DebaterOptimizer: Starting Optimization ===")
        print(f"Current prompt:\n{current_prompt.get_prompt_str()}")
        print(f"Number of results to analyze: {len(results)}")
        print("\n-> Identifying vulnerabilities...")
        vulnerabilities = await self._identify_vulnerabilities(current_prompt, results)
        print(f"Found {len(vulnerabilities)} vulnerabilities:")
        for i, v in enumerate(vulnerabilities, 1):
            print(f"\nVulnerability {i}:")
            print(f"Description: {v.description}")
            print(f"Severity: {v.severity}")
            print(f"Likelihood: {v.likelihood}")
            print(f"Examples: {v.examples}")
        if not vulnerabilities:
            print("No vulnerabilities found, returning current prompt")
            return current_prompt
        print("\n-> Generating defenses...")
        defenses = await self._propose_defenses(current_prompt, vulnerabilities)
        print(f"Generated {len(defenses)} defenses:")
        for i, d in enumerate(defenses, 1):
            print(f"\nDefense {i}:")
            print(f"Changes: {d.prompt_changes}")
            print(f"Addresses: {', '.join(d.addressed_vulnerabilities)}")
            print(f"Tradeoffs: {', '.join(d.tradeoffs)}")
            print(f"Confidence: {d.confidence}")
        if not defenses:
            print("No defenses generated, returning current prompt")
            return current_prompt
        print("\n-> Conducting debate...")
        debate_result = await self._conduct_debate(
            current_prompt, vulnerabilities, defenses
        )
        print("\nDebate results:")
        print(f"Consensus points: {debate_result.consensus_points}")
        print(f"Disagreements: {debate_result.disagreements}")
        print(f"Proposed resolutions: {debate_result.proposed_resolutions}")
        print(f"Confidence ratings: {debate_result.confidence_ratings}")
        print("\n-> Applying consensus changes...")
        improved_prompt = self._apply_consensus_changes(current_prompt, debate_result)
        print("\nFinal improved prompt:")
        print(improved_prompt.get_prompt_str())
        print("\n=== DebaterOptimizer: Optimization Complete ===\n")
        return improved_prompt

    async def _identify_vulnerabilities(
        self,
        prompt: pm_types.PromptWrapper,
        results: list[ExperimentResultRow],
    ) -> list[Vulnerability]:
        print("\n=== Identifying Vulnerabilities ===")
        print(f"Analyzing {len(results)} results for vulnerabilities")
        low_scores = [
            r
            for r in results
            if any(
                eval_result.score < 0.7
                for eval_result in r["evaluation_results"]["results"]
                if eval_result.score is not None
            )
        ]
        print(f"Found {len(low_scores)} examples with low scores (<0.7)")
        if not low_scores:
            print("No low-scoring examples found, prompt appears robust")
            return []
        analysis_prompt = f"""Analyze these examples where the prompt performed poorly:\n\nCurrent Prompt:\n{prompt.get_prompt_str()}\n\nLow-scoring Examples:\n{self._format_examples(low_scores)}\n\nIdentify potential vulnerabilities in the prompt's ability to effectively accomplish its task. Focus on:\n1. Task Understanding: Does the prompt clearly understand what it needs to do?\n2. Output Quality: Are the responses accurate, relevant and well-reasoned?\n3. Edge Cases: What types of inputs cause poor performance?\n4. Consistency: Is the prompt producing consistent results across similar inputs?\n5. Core Functionality: Is the prompt missing key aspects needed for the task?\n\nReturn your analysis in JSON format with an array of vulnerabilities, each containing:\n- description: Clear explanation of the functional vulnerability\n- severity: Float 0-1 indicating how serious the issue is\n- likelihood: Float 0-1 indicating how often this might occur\n- examples: Array of example IDs that demonstrate this vulnerability"""
        print("\nRequesting vulnerability analysis from multiple models...")
        all_vulnerabilities = []
        try:
            tasks = []
            for model_name in self.models:
                model_config = self.model_configs[model_name]
                tasks.append(
                    get_model_response(
                        model_config["model"],
                        analysis_prompt,
                        model_config=model_config,
                    )
                )
            responses = await asyncio.gather(*tasks)
            for response in responses:
                vulnerabilities = self._parse_vulnerabilities(response)
                all_vulnerabilities.extend(vulnerabilities)
            unique_vulnerabilities = []
            seen_descriptions = set()
            for vuln in all_vulnerabilities:
                if vuln.description not in seen_descriptions:
                    seen_descriptions.add(vuln.description)
                    unique_vulnerabilities.append(vuln)
            print(
                f"Successfully parsed {len(unique_vulnerabilities)} unique vulnerabilities from {len(self.models)} models"
            )
            return unique_vulnerabilities
        except Exception as e:
            print(f"Error during vulnerability analysis: {e}")
            print("Full error details:", e.__class__.__name__, str(e))
            return []

    def _format_examples(self, results: list[ExperimentResultRow]) -> str:
        formatted = []
        for i, result in enumerate(results, 1):
            scores = [
                eval_result.score
                for eval_result in result["evaluation_results"]["results"]
                if eval_result.score is not None
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            inputs = result.get("inputs", {})
            outputs = result.get("outputs", {})
            tool_output = ""
            if "tool_calls" in outputs:
                tool_calls = outputs["tool_calls"]
                tool_output = "\nTool Calls:"
                for tool_call in tool_calls:
                    tool_output += f"\n- Tool: {tool_call.get('name', 'unknown')}"
                    tool_output += f"\n  Args: {tool_call.get('args', {})}"
            elif "output" in outputs and isinstance(outputs["output"], dict):
                output_dict = outputs["output"]
                if "tool_calls" in output_dict:
                    tool_calls = output_dict["tool_calls"]
                    tool_output = "\nTool Calls:"
                    for tool_call in tool_calls:
                        tool_output += f"\n- Tool: {tool_call.get('name', 'unknown')}"
                        tool_output += f"\n  Args: {tool_call.get('args', {})}"
            formatted.append(
                f"""
Example {i}:
Input: {inputs}
Output: {outputs.get('output', {}).get('content', outputs.get('content', 'N/A'))}{tool_output}
Score: {avg_score:.4f}
"""
            )
        return "\n".join(formatted)

    async def _propose_defenses(
        self,
        prompt: pm_types.PromptWrapper,
        vulnerabilities: list[Vulnerability],
    ) -> list[Defense]:
        if not vulnerabilities:
            return []
        defense_prompt = f"""Given this prompt and its identified vulnerabilities, propose specific defenses:\n\nCurrent Prompt:\n{prompt.get_prompt_str()}\n\nIdentified Vulnerabilities:\n{self._format_vulnerabilities(vulnerabilities)}\n\nPropose defenses to address these vulnerabilities. Return your suggestions in JSON format with an array of defenses, each containing:\n- prompt_changes: Specific changes to make to the prompt\n- addressed_vulnerabilities: list of vulnerability descriptions being addressed\n- tradeoffs: list of potential tradeoffs or downsides of these changes\n- confidence: Float 0-1 indicating confidence in the defense's effectiveness"""
        print("\nRequesting defense proposals from multiple models...")
        all_defenses = []
        try:
            tasks = []
            for model_name in self.models:
                model_config = self.model_configs[model_name]
                tasks.append(
                    get_model_response(
                        model_config["model"], defense_prompt, model_config=model_config
                    )
                )
            responses = await asyncio.gather(*tasks)
            for response in responses:
                defenses = self._parse_defenses(response)
                all_defenses.extend(defenses)
            unique_defenses = []
            seen_changes = set()
            for defense in all_defenses:
                if defense.prompt_changes not in seen_changes:
                    seen_changes.add(defense.prompt_changes)
                    unique_defenses.append(defense)
            print(
                f"Successfully parsed {len(unique_defenses)} unique defenses from {len(self.models)} models"
            )
            return unique_defenses
        except Exception as e:
            print(f"Error during defense generation: {e}")
            print("Full error details:", e.__class__.__name__, str(e))
            return []

    async def _conduct_debate(
        self,
        prompt: pm_types.PromptWrapper,
        vulnerabilities: list[Vulnerability],
        defenses: list[Defense],
    ) -> DebateRound:
        if not vulnerabilities or not defenses:
            return DebateRound(
                consensus_points=[],
                disagreements=[],
                proposed_resolutions=[],
                confidence_ratings={},
            )
        debate_prompt = f"""Analyze these vulnerabilities and proposed defenses for a prompt:\n\nCurrent Prompt:\n{prompt.get_prompt_str()}\n\nVulnerabilities:\n{self._format_vulnerabilities(vulnerabilities)}\n\nProposed Defenses:\n{self._format_defenses(defenses)}\n\nEvaluate the proposed defenses and identify areas of agreement and disagreement.\nReturn your analysis in JSON format with:\n- consensus_points: list of changes everyone agrees would help\n- disagreements: list of points where there's disagreement\n- proposed_resolutions: list of specific prompt changes to implement (IMPORTANT: Format each resolution as "Action: change" where Action must be one of: Add, Replace, Remove, or Implement. For example: "Add: For emails shorter than 10 characters, return 'Invalid'" or "Replace: old text with new text")\n- confidence_ratings: dict mapping each proposed resolution to a confidence score (0-1)\n\nExample resolution formats:\n- "Add: For emails shorter than 10 characters, return 'Invalid'"\n- "Replace: output only one word with output exactly one of: Positive, Neutral, or Negative"\n- "Remove: optional context section"\n- "Implement: input validation check - if email_content is empty, return 'Neutral'"""
        print("\nConducting debate analysis with multiple models...")
        all_debate_results = []
        try:
            tasks = []
            for model_name in self.models:
                model_config = self.model_configs[model_name]
                tasks.append(
                    get_model_response(
                        model_config["model"], debate_prompt, model_config=model_config
                    )
                )
            responses = await asyncio.gather(*tasks)
            for response in responses:
                debate_data = self._parse_debate_response(response)
                all_debate_results.append(DebateRound(**debate_data))
            final_debate = self._synthesize_debate_responses(
                [asdict(result) for result in all_debate_results]
            )
            print(
                f"Successfully synthesized debate results from {len(self.models)} models"
            )
            return final_debate
        except Exception as e:
            print(f"Error during debate: {e}")
            print("Full error details:", e.__class__.__name__, str(e))
            return DebateRound(
                consensus_points=[],
                disagreements=[],
                proposed_resolutions=[],
                confidence_ratings={},
            )

    def _apply_resolution(self, prompt_str: str, resolution: str) -> str:
        print(f"\nApplying resolution: {resolution}")
        if ":" in resolution:
            _, change = resolution.split(":", 1)
            change = change.strip()
            if "Add" in resolution or "Implement" in resolution:
                if not prompt_str.endswith("\n"):
                    prompt_str += "\n"
                prompt_str += f"\n{change}"
                print(f"Added instruction: {change}")
            elif "Replace" in resolution or "Change" in resolution:
                old_new = change.split(" with ", 1)
                if len(old_new) == 2:
                    old, new = old_new
                    prompt_str = prompt_str.replace(old.strip(), new.strip())
                    print(f"Replaced '{old.strip()}' with '{new.strip()}'")
            elif "Remove" in resolution or "Delete" in resolution:
                to_remove = change.strip()
                prompt_str = prompt_str.replace(to_remove, "")
                print(f"Removed: {to_remove}")
            else:
                lines = prompt_str.split("\n")
                if any("given" in line.lower() for line in lines):
                    for i, line in enumerate(lines):
                        if "given" in line.lower():
                            lines.insert(i + 1, change)
                            break
                else:
                    lines.insert(0, change)
                prompt_str = "\n".join(lines)
                print(f"Applied change: {change}")
            return prompt_str
        print("Could not parse specific changes, keeping original prompt")
        return prompt_str
