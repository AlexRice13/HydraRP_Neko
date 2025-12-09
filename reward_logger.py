"""
Backward-compatible wrapper for reward_logger.

This module maintains backward compatibility with code that imports from reward_logger.
All functionality is now provided by the hydra_rp package.
"""

import warnings

# Import everything from the new library
from hydra_rp.logger import (
    RewardLogger,
    get_logger,
    LLMCallWrapper,
    ScoringLog,
    LLMCallLog,
)

# Create global singleton instance for backward compatibility
reward_logger = get_logger()

# Show deprecation warning on import
warnings.warn(
    "Importing from 'reward_logger' is deprecated. "
    "Please use 'from hydra_rp import get_logger' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'RewardLogger',
    'reward_logger',
    'LLMCallWrapper',
    'ScoringLog',
    'LLMCallLog',
]
