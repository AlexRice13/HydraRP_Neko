"""
Backward-compatible wrapper for orchestrator.

This module maintains backward compatibility with code that imports from orchestrator.
All functionality is now provided by the hydra_rp package.
"""

import warnings

# Import everything from the new library
from hydra_rp.orchestrator import (
    legal_reward_fn,
    aggregate_score_callback,
    get_task_info,
    extract_prompt_text,
    extract_completion_text,
    format_messages_as_history,
    extract_json_like,
    MAX_WORKERS,
    get_registry,
    RewardFunctionRegistry,
)

from hydra_rp.config import (
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_CHAT,
    TASK_TYPE_SCENE,
    TASK_TYPE_MULTI,
    normalize_task_type,
)

# Re-export reward logger for compatibility
from reward_logger import reward_logger

# Show deprecation warning on import
warnings.warn(
    "Importing from 'orchestrator' is deprecated. "
    "Please use 'from hydra_rp import legal_reward_fn' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'legal_reward_fn',
    'aggregate_score_callback',
    'get_task_info',
    'extract_prompt_text',
    'extract_completion_text',
    'format_messages_as_history',
    'extract_json_like',
    'MAX_WORKERS',
    'TASK_TYPE_CODE',
    'TASK_TYPE_IMAGE',
    'TASK_TYPE_CHAT',
    'TASK_TYPE_SCENE',
    'TASK_TYPE_MULTI',
    'normalize_task_type',
    'reward_logger',
    'get_registry',
    'RewardFunctionRegistry',
]
