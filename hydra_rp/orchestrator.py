"""
Orchestrator for reward function evaluation.

This module provides the main entry point for batch evaluation
of model completions with automatic task routing and scoring.
"""

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import (
    TASK_TYPE_CHAT,
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_MULTI,
    TASK_TYPE_SCENE,
    MAX_WORKERS,
    normalize_task_type,
)
from .rewards import (
    calculate_length_penalty,
    calculate_repetition_penalty,
    calculate_thinking_format_penalty,
    reward_code_interpreter,
    reward_daily_chat,
    reward_image_gen,
    reward_multi_turn_chat,
    reward_scene_response,
)
from .logger import get_logger


# ============================================================================
# Task Type Detection
# ============================================================================

def extract_json_like(text: str) -> Optional[Dict]:
    """Extract JSON-like structure from text."""
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if not candidates:
        return None
    candidates = sorted(candidates, key=len)
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def get_task_info(dataset_entry: str, completion: str) -> Tuple[str, Callable]:
    """
    Detect task type from dataset entry and completion.
    
    Args:
        dataset_entry: Dataset entry (may contain JSON)
        completion: Model's completion
    
    Returns:
        Tuple of (task_type, reward_function)
    """
    entry_str = str(dataset_entry)
    data = extract_json_like(entry_str)

    if isinstance(data, dict) and "scenario" in data:
        return TASK_TYPE_SCENE, reward_scene_response

    if isinstance(data, dict) and "prompt" in data and "scenario" not in data:
        return TASK_TYPE_IMAGE, reward_image_gen

    code_patterns = [
        r"(使用|请用|运行|执行|计算)[^，。]*code\s*interpreter",
        r"<code_interpreter\b",
    ]
    for pat in code_patterns:
        if re.search(pat, entry_str, re.IGNORECASE | re.DOTALL):
            return TASK_TYPE_CODE, reward_code_interpreter

    return TASK_TYPE_CHAT, reward_daily_chat


# ============================================================================
# Message Extraction
# ============================================================================

def extract_prompt_text(messages: List[Dict[str, str]]) -> str:
    """Extract user prompt from messages list."""
    if not messages or not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def extract_completion_text(completion: Any) -> str:
    """Extract completion text from various formats."""
    if not completion:
        return ""

    if isinstance(completion, list):
        if len(completion) == 0:
            return ""
        for msg in completion:
            if isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    return content if isinstance(content, str) else str(content)
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            content = last_msg.get("content", "")
            return content if isinstance(content, str) else str(content)
        elif isinstance(last_msg, str):
            return last_msg
        return str(last_msg)

    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)

    if isinstance(completion, str):
        return completion

    return str(completion)


def format_messages_as_history(messages: List[Dict[str, str]]) -> str:
    """Format messages list as dialogue history."""
    if not messages or not isinstance(messages, list):
        return ""
    history_parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                history_parts.append(f"[System]: {content}")
            elif role == "user":
                history_parts.append(f"[User]: {content}")
            elif role == "assistant":
                history_parts.append(f"[Assistant]: {content}")
    return "\n".join(history_parts)


# ============================================================================
# Scoring Functions
# ============================================================================

def aggregate_score_callback(
    prompt_text: str,
    completion_text: str,
    task_type: str
) -> Dict[str, Any]:
    """
    Aggregate scoring with penalties.
    
    Args:
        prompt_text: User prompt text
        completion_text: Model completion text
        task_type: Task type identifier
    
    Returns:
        Dict with scoring breakdown
    """
    default_failure_score = 0.0
    default_error_result = {
        "task_type": "ERROR_CRASH",
        "raw_reward": default_failure_score,
        "penalty_repetition": default_failure_score,
        "penalty_length": default_failure_score,
        "penalty_thinking": default_failure_score,
        "final_score": default_failure_score,
    }

    try:
        logger = get_logger()
        with logger.scoring_context(prompt_text, completion_text) as log:
            internal_task_type = normalize_task_type(task_type)
            log.task_type = internal_task_type

            reward_func_map = {
                TASK_TYPE_CHAT: reward_daily_chat,
                TASK_TYPE_SCENE: reward_scene_response,
                TASK_TYPE_IMAGE: reward_image_gen,
                TASK_TYPE_CODE: reward_code_interpreter,
                TASK_TYPE_MULTI: reward_multi_turn_chat,
            }
            reward_func = reward_func_map.get(internal_task_type, reward_daily_chat)
            log.reward_function = reward_func.__name__

            raw_reward = reward_func(completion_text, prompt_text)
            log.raw_reward = raw_reward

            penalty_rep = calculate_repetition_penalty(
                completion_text, internal_task_type
            )
            penalty_len = calculate_length_penalty(
                completion_text, internal_task_type
            )
            penalty_thk = calculate_thinking_format_penalty(
                completion_text, internal_task_type
            )

            log.penalty_repetition = penalty_rep
            log.penalty_length = penalty_len
            log.penalty_thinking = penalty_thk
            log.total_penalty = penalty_rep + penalty_len + penalty_thk

            final_score = float(raw_reward - log.total_penalty)
            log.final_score = final_score

            return {
                "task_type": internal_task_type,
                "raw_reward": round(raw_reward, 3),
                "penalty_repetition": round(penalty_rep, 3),
                "penalty_length": round(penalty_len, 3),
                "penalty_thinking": round(penalty_thk, 3),
                "final_score": round(final_score, 3),
            }

    except Exception as e:
        logging.error(f"CRITICAL SCORING FAILURE. Error: {e}", exc_info=True)
        return default_error_result


def _process_single_sample(args: Tuple[Any, Any, str]) -> float:
    """
    Process a single sample for scoring.
    
    Args:
        args: Tuple of (messages, completion, task_type)
    
    Returns:
        Final score as float
    """
    messages, completion, task_type = args
    internal_task_type = normalize_task_type(task_type)

    if internal_task_type == TASK_TYPE_MULTI:
        prompt_text = format_messages_as_history(messages)
    else:
        prompt_text = extract_prompt_text(messages)

    completion_text = extract_completion_text(completion)

    result = aggregate_score_callback(
        prompt_text=prompt_text,
        completion_text=completion_text,
        task_type=task_type,
    )

    try:
        return float(result["final_score"])
    except (KeyError, ValueError, TypeError):
        return 0.0


# ============================================================================
# Main Reward Function
# ============================================================================

def legal_reward_fn(
    prompts: Optional[List] = None,
    completions: Optional[List] = None,
    **kwargs,
) -> List[float]:
    """
    Batch reward function for model evaluation.
    
    This is the main entry point for evaluating model completions.
    Supports concurrent processing of multiple samples.
    
    Args:
        prompts: List of prompt messages
        completions: List of model completions
        **kwargs: Additional arguments including:
            - task_type: List of task type identifiers
    
    Returns:
        List of final scores (floats)
    
    Example:
        >>> from hydra_rp import legal_reward_fn
        >>> prompts = [[{"role": "user", "content": "Hello!"}]]
        >>> completions = [{"content": "(微笑) 你好！"}]
        >>> scores = legal_reward_fn(prompts, completions, task_type=["chat"])
    """
    if prompts is None:
        prompts = kwargs.get("prompts", [])
    if completions is None:
        completions = kwargs.get("completions", [])

    task_types = kwargs.get("task_type", [])

    batch_size = len(prompts)

    if batch_size == 0:
        logging.warning("legal_reward_fn: Empty prompts list")
        return []

    if len(completions) != batch_size:
        logging.warning(
            f"legal_reward_fn: completions length ({len(completions)}) "
            f"doesn't match prompts length ({batch_size})"
        )
        if len(completions) < batch_size:
            completions = completions + [[]] * (batch_size - len(completions))
        else:
            completions = completions[:batch_size]

    if len(task_types) < batch_size:
        default_type = TASK_TYPE_CHAT
        task_types = list(task_types) + [default_type] * (
            batch_size - len(task_types)
        )

    input_tuples: List[Tuple[Any, Any, str]] = []
    for i in range(batch_size):
        messages = prompts[i]
        completion = completions[i]
        task_type = task_types[i] if i < len(task_types) else TASK_TYPE_CHAT
        input_tuples.append((messages, completion, task_type))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rewards = list(executor.map(_process_single_sample, input_tuples))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"legal_reward_fn: Processed {batch_size} samples")
        logging.debug(f"  - Task type distribution: {Counter(task_types)}")
        if rewards:
            logging.debug(
                f"  - Reward stats: min={min(rewards):.3f}, "
                f"max={max(rewards):.3f}, "
                f"mean={sum(rewards)/len(rewards):.3f}"
            )

    return rewards


# ============================================================================
# Reward Function Registry
# ============================================================================

class RewardFunctionRegistry:
    """
    Registry for managing custom reward functions.
    
    Allows extending the system with new task types and reward functions.
    """
    
    def __init__(self):
        self._reward_functions = {
            TASK_TYPE_CHAT: reward_daily_chat,
            TASK_TYPE_SCENE: reward_scene_response,
            TASK_TYPE_IMAGE: reward_image_gen,
            TASK_TYPE_CODE: reward_code_interpreter,
            TASK_TYPE_MULTI: reward_multi_turn_chat,
        }
    
    def register(self, task_type: str, reward_func: Callable) -> None:
        """
        Register a custom reward function for a task type.
        
        Args:
            task_type: Task type identifier
            reward_func: Reward function with signature (completion, prompt) -> float
        """
        self._reward_functions[task_type] = reward_func
    
    def get(self, task_type: str) -> Callable:
        """Get reward function for task type."""
        return self._reward_functions.get(task_type, reward_daily_chat)
    
    def list_task_types(self) -> List[str]:
        """List all registered task types."""
        return list(self._reward_functions.keys())


# Global registry
_registry = RewardFunctionRegistry()


def get_registry() -> RewardFunctionRegistry:
    """Get the global reward function registry."""
    return _registry
