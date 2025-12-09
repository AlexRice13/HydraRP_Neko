"""
HydraRP - Reward function library for LLM evaluation.

This library provides a comprehensive system for evaluating model completions
across different task types with automatic scoring, penalties, and logging.

Main Features:
- Multi-task type support (chat, code, image, scene, multi-turn)
- Automatic task detection and routing
- LLM-based judging with detailed logging
- Penalty system for quality issues
- Extensible architecture

Quick Start:
    >>> from hydra_rp import legal_reward_fn
    >>> prompts = [[{"role": "user", "content": "Hello!"}]]
    >>> completions = [{"content": "(微笑) 你好！"}]
    >>> scores = legal_reward_fn(prompts, completions, task_type=["chat"])

Advanced Usage:
    >>> from hydra_rp import get_registry, reward_daily_chat
    >>> registry = get_registry()
    >>> registry.register("custom_task", my_custom_reward_func)
"""

__version__ = "1.0.0"

# Core orchestration
from .orchestrator import (
    legal_reward_fn,
    aggregate_score_callback,
    get_registry,
    RewardFunctionRegistry,
)

# Reward functions
from .rewards import (
    reward_code_interpreter,
    reward_daily_chat,
    reward_image_gen,
    reward_multi_turn_chat,
    reward_scene_response,
    calculate_repetition_penalty,
    calculate_length_penalty,
    calculate_thinking_format_penalty,
)

# LLM client
from .llm_client import (
    JudgeLLMClient,
    get_client,
    call_judge_llm,
    extract_score_from_response,
)

# Logger
from .logger import (
    RewardLogger,
    get_logger,
    LLMCallWrapper,
    ScoringLog,
    LLMCallLog,
)

# Configuration
from .config import (
    TASK_TYPE_CHAT,
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_SCENE,
    TASK_TYPE_MULTI,
    normalize_task_type,
)

__all__ = [
    # Main entry point
    "legal_reward_fn",
    
    # Reward functions
    "reward_code_interpreter",
    "reward_daily_chat",
    "reward_image_gen",
    "reward_multi_turn_chat",
    "reward_scene_response",
    
    # Penalties
    "calculate_repetition_penalty",
    "calculate_length_penalty",
    "calculate_thinking_format_penalty",
    
    # Orchestration
    "aggregate_score_callback",
    "get_registry",
    "RewardFunctionRegistry",
    
    # LLM client
    "JudgeLLMClient",
    "get_client",
    "call_judge_llm",
    "extract_score_from_response",
    
    # Logger
    "RewardLogger",
    "get_logger",
    "LLMCallWrapper",
    "ScoringLog",
    "LLMCallLog",
    
    # Task types
    "TASK_TYPE_CHAT",
    "TASK_TYPE_CODE",
    "TASK_TYPE_IMAGE",
    "TASK_TYPE_SCENE",
    "TASK_TYPE_MULTI",
    "normalize_task_type",
]
