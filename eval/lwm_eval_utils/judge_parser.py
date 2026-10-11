"""
AgentWorldBench Judge Output Parser

Robust JSON extraction from LLM Judge output with multiple strategies.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from .task_configs import SCORE_DIMENSIONS, TASK_CONFIGS
from .output_parser import _remove_thinking_tags

# Base directory for resolving relative paths in TASK_CONFIGS
# Prompts live at repo root: prompts/{domain}/judge_system_prompt.txt
# This file is at eval/lwm_eval_utils/judge_parser.py, so repo root is ../../
_UTILS_DIR = Path(__file__).absolute().parent.parent.parent

logger = logging.getLogger(__name__)


# =============================================================================
# Main Parser Function
# =============================================================================

def parse_judge_output(raw_output: str, response_tag: str) -> Dict[str, Any]:
    """Parse LLM Judge output and extract scores.
    
    Args:
        raw_output: The raw text output from the judge model.
        response_tag: The tag wrapping the final JSON output (default: "final_evaluation").
    """
    try:
        cleaned = raw_output.strip() if raw_output else ""
        if not cleaned:
            logger.warning("LLM Judge returned empty output")
            return _get_error_result("Empty output", raw_output)

        # Remove thinking tags
        cleaned = _remove_thinking_tags(cleaned, response_tag=response_tag)

        # Extract content from response tag if present
        json_content = _extract_tagged_content(cleaned, response_tag)
        
        # If no tagged content found, use the whole cleaned text
        text_to_parse = json_content if json_content else cleaned

        # Try extraction strategies in order on the target text
        json_str = (
            _extract_from_markdown(text_to_parse) or
            _extract_best_json_object(text_to_parse) or
            _extract_last_json(text_to_parse)
        )
        
        # If that failed, and we had a tag, maybe the tag content itself IS the JSON (without markdown blocks)
        if not json_str and json_content:
             json_str = (
                _extract_best_json_object(json_content) or
                _extract_last_json(json_content)
            )

        if not json_str:
            logger.warning("No valid JSON object found in LLM Judge output")
            logger.debug(f"Raw output (first 500 chars): {raw_output[:500] if raw_output else ''}")
            return _get_error_result("No JSON object found", raw_output)

        # Repair and parse JSON
        json_str = _repair_json(json_str)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            repaired = _extract_scores_only(json_str)
            if repaired:
                result = json.loads(repaired)
            else:
                raise e

        # Extract and validate scores
        scores = _extract_scores(result)
        if not scores or any(value < 1 for value in scores.values()):
            logger.warning("Missing or invalid 'scores' field in LLM Judge output")
            logger.debug(f"Available keys: {list(result.keys())}")
            return _get_error_result(f"Invalid scores. Keys: {list(result.keys())}", raw_output)

        return {
            "strengths": _to_list(result.get("strengths", [])),
            "weaknesses": _to_list(result.get("weaknesses", [])),
            "scores": scores,
            "total_score": sum(scores.values()) / len(scores),
            "success": True,
            "judge_raw_output": raw_output,
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error parsing LLM Judge output: {e}")
        logger.debug(f"Raw output (first 500 chars):\n{raw_output[:500] if raw_output else ''}")
        return _get_error_result(f"JSON error: {e}", raw_output)
    except Exception as e:
        logger.warning(f"Unexpected error parsing LLM Judge output: {e}")
        logger.debug(f"Raw output (first 500 chars):\n{raw_output[:500] if raw_output else ''}")
        return _get_error_result(f"Error: {e}", raw_output)


def _extract_tagged_content(text: str, tag: str) -> Optional[str]:
    """Extract content from the LAST valid block of the given tag."""
    if not tag:
        return None
        
    start_pattern = rf"<{re.escape(tag)}>"
    start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
    
    if not start_matches:
        return None
    
    # Take the last start tag
    last_start_match = start_matches[-1]
    start_pos = last_start_match.end()
    
    # Look for a closing tag AFTER the start tag
    close_pattern = rf"</{re.escape(tag)}>"
    close_match = re.search(close_pattern, text[start_pos:], re.IGNORECASE)
    
    if close_match:
        end_pos = start_pos + close_match.start()
        return text[start_pos:end_pos].strip()
    else:
        # No closing tag, return everything until end
        return text[start_pos:].strip()


# =============================================================================
# JSON Extraction Strategies
# =============================================================================

def _extract_from_markdown(text: str) -> Optional[str]:
    """Extract JSON from ```json``` code blocks (prefer last with 'scores')."""
    matches = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text))
    if not matches:
        return None
    
    # Prefer last block with "scores"
    for m in reversed(matches):
        c = m.group(1).strip()
        if c.startswith('{') and c.endswith('}') and '"scores"' in c:
            return c
    
    # Fallback to last valid JSON block
    for m in reversed(matches):
        c = m.group(1).strip()
        if c.startswith('{') and c.endswith('}'):
            return c
    return None


def _extract_best_json_object(text: str) -> Optional[str]:
    """Extract all JSON objects, return last one with 'scores'."""
    objects = []
    pos = 0
    while True:
        start = text.find('{', pos)
        if start == -1:
            break
        obj = _match_braces_forward(text, start)
        if obj:
            objects.append(obj)
            pos = start + len(obj)
        else:
            pos = start + 1
    
    if not objects:
        return None
    
    # Prefer object with "scores"
    for obj in reversed(objects):
        if '"scores"' in obj:
            return obj
    return objects[-1]


def _extract_last_json(text: str) -> Optional[str]:
    """Extract last JSON object by matching braces backwards."""
    end = text.rfind('}')
    if end == -1:
        return None
    
    depth, in_str, i = 0, False, end
    while i >= 0:
        c = text[i]
        if c == '"' and (i == 0 or text[i-1] != '\\'):
            in_str = not in_str
        if not in_str:
            if c == '}':
                depth += 1
            elif c == '{':
                depth -= 1
                if depth == 0:
                    return text[i:end+1]
        i -= 1
    return None


def _match_braces_forward(text: str, start: int) -> Optional[str]:
    """Match braces forward from start position."""
    if start >= len(text) or text[start] != '{':
        return None
    
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


# =============================================================================
# JSON Repair
# =============================================================================

def _repair_json(s: str) -> str:
    """Fix common JSON issues: trailing commas, single quotes, unquoted keys."""
    if not s:
        return s
    s = re.sub(r',(\s*[}\]])', r'\1', s)  # trailing commas
    s = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', s)  # single-quoted keys
    s = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', s)  # unquoted keys
    return s


def _extract_scores_only(s: str) -> Optional[str]:
    """Last resort: extract just scores and build minimal JSON."""
    m = re.search(r'"scores"\s*:\s*\{[^}]+\}', s)
    if m:
        return '{"strengths":[],"weaknesses":[],' + m.group(0) + '}'
    return None


# =============================================================================
# Score Extraction and Validation
# =============================================================================

def _extract_scores(result: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract scores dict, handling int/float/str values."""
    if "scores" not in result or not isinstance(result["scores"], dict):
        return None
    
    scores = {}
    for dim in SCORE_DIMENSIONS:
        v = result["scores"].get(dim, 0)
        if isinstance(v, int):
            scores[dim] = v
        elif isinstance(v, float):
            scores[dim] = int(round(v))
        elif isinstance(v, str):
            try:
                scores[dim] = int(v.split('/')[0].strip())
            except (ValueError, TypeError, AttributeError):
                scores[dim] = 0
        else:
            scores[dim] = 0
        # Clamp to [1, 5] or keep 0 for error
        if scores[dim] > 0:
            scores[dim] = max(1, min(5, scores[dim]))
    return scores


def _to_list(value: Any) -> List[str]:
    """Convert value to list of strings."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value] if value else []
    return []


# =============================================================================
# Error Handling
# =============================================================================

def _get_error_result(msg: str, raw: str = "") -> Dict[str, Any]:
    """Return standardized error result."""
    return {
        "error_message": msg,
        "strengths": [],
        "weaknesses": [],
        "scores": {d: 0 for d in SCORE_DIMENSIONS},
        "total_score": 0.0,
        "success": False,
        "judge_raw_output": raw,
    }


# =============================================================================
# Choice Answer Extraction (for Turing Test / Ref Answer tasks)
# =============================================================================

def extract_choice_answer(raw_output: str) -> Dict[str, Any]:
    """Extract choice answer (A or B) from Judge output.
    
    Looks for \boxed{A} or \boxed{B} pattern in the output.
    Also handles variations like \\boxed{A}, \boxed{ A }, etc.
    
    Args:
        raw_output: Raw LLM Judge output string
        
    Returns:
        Dict containing:
        - choice: "A" | "B" | None (extracted choice)
        - reward: 1.0 if A, 0.0 if B, 0.0 if None
        - success: bool (whether extraction succeeded)
        - judge_raw_output: str (original output)
        - error_message: str (only if failed)
    """
    if not raw_output or not raw_output.strip():
        return {
            "choice": None,
            "reward": 0.0,
            "success": False,
            "judge_raw_output": raw_output or "",
            "error_message": "Empty output",
        }
    
    cleaned = raw_output.strip()
    
    # Remove thinking tags first
    cleaned = _remove_thinking_tags(cleaned, "")
    
    # Pattern to match \boxed{A} or \boxed{B} with various formats
    # Handles: \boxed{A}, \\boxed{A}, \boxed{ A }, \boxed{a}, etc.
    patterns = [
        r"\\boxed\{\s*([AaBb])\s*\}",  # Standard LaTeX: \boxed{A}
        r"\\\\boxed\{\s*([AaBb])\s*\}",  # Escaped backslash: \\boxed{A}
        r"\$\\boxed\{\s*([AaBb])\s*\}\$",  # With dollar signs: $\boxed{A}$
        r"boxed\{\s*([AaBb])\s*\}",  # Without backslash: boxed{A}
    ]
    
    choice = None
    
    # Try each pattern, prefer the last match (final answer)
    for pattern in patterns:
        matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        if matches:
            # Take the last match as the final answer
            choice = matches[-1].group(1).upper()
            break
    
    if choice is None:
        # Fallback: look for explicit statements like "my answer is A" or "I choose B"
        fallback_patterns = [
            r"(?:answer|choice|select|choose|pick)\s*(?:is|:)?\s*\**\s*([AaBb])\b",
            r"\b([AaBb])\s*(?:is|appears|seems)\s+(?:more\s+)?(?:realistic|correct|better)",
            r"(?:Response|Option)\s+([AaBb])\s+is\s+(?:the\s+)?(?:real|correct|better)",
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                choice = match.group(1).upper()
                break
    
    if choice in ("A", "B"):
        return {
            "choice": choice,
            "reward": 1.0 if choice == "A" else 0.0,
            "success": True,
            "judge_raw_output": raw_output,
        }
    else:
        logger.warning(f"Failed to extract choice from Judge output")
        logger.debug(f"Raw output (first 500 chars): {raw_output[:500]}")
        return {
            "choice": None,
            "reward": 0.0,
            "success": False,
            "judge_raw_output": raw_output,
            "error_message": "Could not extract A or B choice",
        }


# =============================================================================
# Prompt Loading
# =============================================================================

def _load_prompts_by_config_key(config_key: str) -> Dict[str, str]:
    """Load prompts for all subtasks using a path field from TASK_CONFIGS.

    Args:
        config_key: The key in each subtask config that holds the relative prompt path.

    Returns:
        Dict mapping subtask name to its prompt string.
    """
    prompts = {}
    for subtask, config in TASK_CONFIGS.items():
        prompt_path = config.get(config_key)
        if not prompt_path:
            logger.warning(f"No {config_key} configured for subtask: {subtask}")
            prompts[subtask] = ""
            continue

        full_path = _UTILS_DIR / prompt_path
        if full_path.exists():
            prompts[subtask] = full_path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Prompt file not found: {full_path}")
            prompts[subtask] = ""
    return prompts


def load_judge_system_prompts() -> Dict[str, str]:
    """Load judge system prompts for all subtasks."""
    return _load_prompts_by_config_key("judge_system_prompt_path")
