"""
Backward-compatible wrapper for llm_client.

This module maintains backward compatibility with code that imports from llm_client.
All functionality is now provided by the hydra_rp package.
"""

import warnings

# Import everything from the new library
from hydra_rp.config import (
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_CHAT,
    TASK_TYPE_SCENE,
    TASK_TYPE_MULTI,
    TASK_TYPE_MAPPING,
    normalize_task_type,
)
from hydra_rp.llm_client import (
    call_judge_llm,
    extract_score_from_response,
    get_client,
)

# Maintain backward compatibility with global client
judge_client = get_client().client

# Show deprecation warning on import
warnings.warn(
    "Importing from 'llm_client' is deprecated. "
    "Please use 'from hydra_rp import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'TASK_TYPE_CODE',
    'TASK_TYPE_IMAGE',
    'TASK_TYPE_CHAT',
    'TASK_TYPE_SCENE',
    'TASK_TYPE_MULTI',
    'TASK_TYPE_MAPPING',
    'normalize_task_type',
    'call_judge_llm',
    'extract_score_from_response',
    'judge_client',
]