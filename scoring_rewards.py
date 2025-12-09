"""
Backward-compatible wrapper for scoring_rewards.

This module maintains backward compatibility with code that imports from scoring_rewards.
All functionality is now provided by the hydra_rp package.
"""

import warnings

# Import everything from the new library
from hydra_rp.rewards import (
    reward_code_interpreter,
    reward_daily_chat,
    reward_image_gen,
    reward_multi_turn_chat,
    reward_scene_response,
    calculate_repetition_penalty,
    calculate_length_penalty,
    calculate_thinking_format_penalty,
    execute_code_safely,
    KAOMOJI_CHARS,
    is_allowed_char,
    detect_gibberish_chars,
    detect_thought_collapse,
    detect_response_quality,
    extract_think_and_response,
)

# Show deprecation warning on import
warnings.warn(
    "Importing from 'scoring_rewards' is deprecated. "
    "Please use 'from hydra_rp import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'reward_code_interpreter',
    'reward_daily_chat',
    'reward_image_gen',
    'reward_multi_turn_chat',
    'reward_scene_response',
    'calculate_repetition_penalty',
    'calculate_length_penalty',
    'calculate_thinking_format_penalty',
    'execute_code_safely',
    'KAOMOJI_CHARS',
    'is_allowed_char',
    'detect_gibberish_chars',
    'detect_thought_collapse',
    'detect_response_quality',
    'extract_think_and_response',
]
