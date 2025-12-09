import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_client import (
    TASK_TYPE_CHAT,
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_MULTI,
    TASK_TYPE_SCENE,
    normalize_task_type,
)
from scoring_rewards import (
    calculate_length_penalty,
    calculate_repetition_penalty,
    calculate_thinking_format_penalty,
    reward_code_interpreter,
    reward_daily_chat,
    reward_image_gen,
    reward_multi_turn_chat,
    reward_scene_response,
)

MAX_WORKERS = 32


def extract_json_like(text: str):
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


def aggregate_score_callback(prompt_text: str, completion_text: str, task_type: str) -> Dict[str, Any]:
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
        with reward_logger.scoring_context(prompt_text, completion_text) as log:
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

            penalty_rep = calculate_repetition_penalty(completion_text, internal_task_type)
            penalty_len = calculate_length_penalty(completion_text, internal_task_type)
            penalty_thk = calculate_thinking_format_penalty(completion_text, internal_task_type)

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


def extract_prompt_text(messages: List[Dict[str, str]]) -> str:
    if not messages or not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def extract_completion_text(completion: Any) -> str:
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


def _process_single_sample(args: Tuple[Any, Any, str]) -> float:
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


def legal_reward_fn(
    prompts: Optional[List] = None,
    completions: Optional[List] = None,
    **kwargs,
) -> List[float]:
    if prompts is None:
        prompts = kwargs.get("prompts", [])
    if completions is None:
        completions = kwargs.get("completions", [])

    task_types = kwargs.get("task_type", [])

    batch_size = len(prompts)

    if batch_size == 0:
        logging.warning("legal_reward_fn: 空的 prompts 列表")
        return []

    if len(completions) != batch_size:
        logging.warning(
            f"legal_reward_fn: completions 长度 ({len(completions)}) 与 prompts 长度 ({batch_size}) 不匹配"
        )
        if len(completions) < batch_size:
            completions = completions + [[]] * (batch_size - len(completions))
        else:
            completions = completions[:batch_size]

    if len(task_types) < batch_size:
        default_type = TASK_TYPE_CHAT
        task_types = list(task_types) + [default_type] * (batch_size - len(task_types))

    input_tuples: List[Tuple[Any, Any, str]] = []
    for i in range(batch_size):
        messages = prompts[i]
        completion = completions[i]
        task_type = task_types[i] if i < len(task_types) else TASK_TYPE_CHAT
        input_tuples.append((messages, completion, task_type))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rewards = list(executor.map(_process_single_sample, input_tuples))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"legal_reward_fn: 处理了 {batch_size} 个样本")
        logging.debug(f"  - 任务类型分布: {Counter(task_types)}")
        logging.debug(
            f"  - 奖励统计: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={sum(rewards)/len(rewards):.3f}"
        )

    return rewards