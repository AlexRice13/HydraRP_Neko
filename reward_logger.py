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
# Note: This uses lazy initialization - logger is created when first accessed
_logger_instance = None

def _get_logger_instance():
    """Get or create the logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = get_logger()
    return _logger_instance

# Make reward_logger auto-initialize on first access
class _LazyLogger:
    """Lazy initialization wrapper for reward_logger."""
    def __getattr__(self, name):
        return getattr(_get_logger_instance(), name)

reward_logger = _LazyLogger()

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
