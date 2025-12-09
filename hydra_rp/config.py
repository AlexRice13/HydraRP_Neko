"""
Configuration module for HydraRP library.

This module contains all configuration constants and environment variables
used throughout the library.
"""

import os
from typing import Optional, Dict

# ============================================================================
# Runtime Configuration Storage
# ============================================================================

# Storage for programmatically set configuration values
# This is useful for Jupyter notebooks where setting environment variables is inconvenient
_runtime_config: Dict[str, any] = {}


def set_config(
    judge_api_endpoint: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    log_dir: Optional[str] = None,
    logging_enabled: Optional[bool] = None,
    max_text_len: Optional[int] = None,
    code_judge_prompt: Optional[str] = None,
    chat_judge_prompt: Optional[str] = None,
    multi_turn_judge_prompt: Optional[str] = None,
    sac_judge_prompt: Optional[str] = None,
) -> None:
    """
    Set configuration values programmatically for Jupyter notebook usage.
    
    This is an alternative to setting environment variables, especially useful
    in Jupyter notebooks where environment variables are less convenient.
    
    Args:
        judge_api_endpoint: API endpoint for judge LLM
        judge_api_key: API key for judge LLM
        judge_model_name: Model name for judge LLM
        log_dir: Directory for log files
        logging_enabled: Whether logging is enabled
        max_text_len: Maximum text length for logging
        code_judge_prompt: Custom prompt for code evaluation
        chat_judge_prompt: Custom prompt for chat evaluation
        multi_turn_judge_prompt: Custom prompt for multi-turn chat evaluation
        sac_judge_prompt: Custom prompt for scene/scenario evaluation
    
    Example:
        >>> from hydra_rp import set_config
        >>> set_config(
        ...     judge_api_key="sk-...",
        ...     judge_model_name="gpt-4",
        ...     log_dir="./my_logs"
        ... )
    """
    global _runtime_config
    
    if judge_api_endpoint is not None:
        _runtime_config['JUDGE_API_ENDPOINT'] = judge_api_endpoint
    if judge_api_key is not None:
        _runtime_config['JUDGE_API_KEY'] = judge_api_key
    if judge_model_name is not None:
        _runtime_config['JUDGE_MODEL_NAME'] = judge_model_name
    if log_dir is not None:
        _runtime_config['HYDRA_LOG_DIR'] = log_dir
    if logging_enabled is not None:
        _runtime_config['HYDRA_LOGGING_ENABLED'] = str(logging_enabled).lower()
    if max_text_len is not None:
        _runtime_config['HYDRA_MAX_TEXT_LEN'] = str(max_text_len)
    if code_judge_prompt is not None:
        _runtime_config['CODE_JUDGE_PROMPT'] = code_judge_prompt
    if chat_judge_prompt is not None:
        _runtime_config['CHAT_JUDGE_PROMPT'] = chat_judge_prompt
    if multi_turn_judge_prompt is not None:
        _runtime_config['MULTI_TURN_JUDGE_PROMPT'] = multi_turn_judge_prompt
    if sac_judge_prompt is not None:
        _runtime_config['SAC_JUDGE_PROMPT'] = sac_judge_prompt


def get_config(key: str, default: any = None) -> any:
    """
    Get a configuration value.
    
    Checks runtime config first, then environment variables, then returns default.
    
    Args:
        key: Configuration key
        default: Default value if not found
    
    Returns:
        Configuration value
    """
    # Check runtime config first
    if key in _runtime_config:
        return _runtime_config[key]
    # Then check environment variables
    return os.getenv(key, default)


def clear_config() -> None:
    """Clear all programmatically set configuration values."""
    global _runtime_config
    _runtime_config = {}


# ============================================================================
# Task Type Constants
# ============================================================================

TASK_TYPE_CODE = "code_interpreter"
TASK_TYPE_IMAGE = "image_gen"
TASK_TYPE_CHAT = "daily_chat"
TASK_TYPE_SCENE = "scene_response"
TASK_TYPE_MULTI = "multi_turn_chat"

TASK_TYPE_MAPPING = {
    "coding": TASK_TYPE_CODE,
    "code": TASK_TYPE_CODE,
    "code_interpreter": TASK_TYPE_CODE,
    "image_gen": TASK_TYPE_IMAGE,
    "image": TASK_TYPE_IMAGE,
    "chat": TASK_TYPE_CHAT,
    "daily_chat": TASK_TYPE_CHAT,
    "sac": TASK_TYPE_SCENE,
    "scene": TASK_TYPE_SCENE,
    "scene_response": TASK_TYPE_SCENE,
    "multi_turn": TASK_TYPE_MULTI,
    "multi_chat": TASK_TYPE_MULTI,
    "history_chat": TASK_TYPE_MULTI,
    "story_mode": TASK_TYPE_MULTI,
    "continuous": TASK_TYPE_MULTI,
}


def normalize_task_type(task_type: str) -> str:
    """Normalize task type string to standard format."""
    if not task_type:
        return TASK_TYPE_CHAT
    return TASK_TYPE_MAPPING.get(task_type.lower().strip(), TASK_TYPE_CHAT)


# ============================================================================
# LLM Judge Configuration
# ============================================================================

def get_judge_api_endpoint() -> str:
    """Get judge API endpoint from runtime config or environment."""
    return get_config("JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")


def get_judge_api_key() -> str:
    """Get judge API key from runtime config or environment."""
    return get_config("JUDGE_API_KEY", "")


def get_judge_model_name() -> str:
    """Get judge model name from runtime config or environment."""
    return get_config("JUDGE_MODEL_NAME", "gpt-4")


# ============================================================================
# Judge Prompts
# ============================================================================

# Default prompts
_DEFAULT_CODE_JUDGE_PROMPT = """You are a code execution and correctness evaluator.

Given:
- User's question/request
- Model's code output
- Code execution result

Evaluate whether the code correctly solves the problem.

Output format:
Score: [0.0-1.0]
Analysis: [brief explanation]

Scoring criteria:
- 1.0: Perfect solution, correct output
- 0.7-0.9: Mostly correct, minor issues
- 0.4-0.6: Partially correct
- 0.0-0.3: Incorrect or failed
"""

_DEFAULT_CHAT_JUDGE_PROMPT = """You are evaluating a chat response quality.

Evaluate the response based on:
- Relevance to user's message
- Natural conversation flow
- Character consistency (cat girl persona)
- Appropriate use of actions in parentheses

Output format:
Score: [0.0-1.0]
Analysis: [brief explanation]

Scoring criteria:
- 1.0: Excellent response, very engaging
- 0.7-0.9: Good response, appropriate
- 0.4-0.6: Acceptable but could be better
- 0.0-0.3: Poor or off-topic response
"""

_DEFAULT_MULTI_TURN_JUDGE_PROMPT = """You are evaluating multi-turn dialogue quality.

Given the dialogue history and current response, evaluate:
- Context awareness and continuity
- Character consistency across turns
- Natural progression of conversation
- Appropriate reactions to previous messages

Output format:
Score: [0.0-1.0]
Analysis: [brief explanation]

Scoring criteria:
- 1.0: Perfect continuity and coherence
- 0.7-0.9: Good context awareness
- 0.4-0.6: Some context issues
- 0.0-0.3: Poor context handling
"""

_DEFAULT_SAC_JUDGE_PROMPT = """You are evaluating a scene/scenario response.

Evaluate the response based on:
- Scene immersion and atmosphere
- Character actions and emotions
- Narrative quality
- Consistency with scenario

Output format:
Score: [0.0-1.0]
Analysis: [brief explanation]

Scoring criteria:
- 1.0: Excellent scene response, highly immersive
- 0.7-0.9: Good scene engagement
- 0.4-0.6: Acceptable scene response
- 0.0-0.3: Poor or generic response
"""


# Getter functions for prompts (check runtime config first)
def get_code_judge_prompt() -> str:
    """Get code judge prompt from runtime config or return default."""
    return get_config("CODE_JUDGE_PROMPT", _DEFAULT_CODE_JUDGE_PROMPT)


def get_chat_judge_prompt() -> str:
    """Get chat judge prompt from runtime config or return default."""
    return get_config("CHAT_JUDGE_PROMPT", _DEFAULT_CHAT_JUDGE_PROMPT)


def get_multi_turn_judge_prompt() -> str:
    """Get multi-turn judge prompt from runtime config or return default."""
    return get_config("MULTI_TURN_JUDGE_PROMPT", _DEFAULT_MULTI_TURN_JUDGE_PROMPT)


def get_sac_judge_prompt() -> str:
    """Get scene/scenario judge prompt from runtime config or return default."""
    return get_config("SAC_JUDGE_PROMPT", _DEFAULT_SAC_JUDGE_PROMPT)


# Backward-compatible aliases - use defaults directly for now
CODE_JUDGE_PRMPT = CODE_JUDGE_PROMPT = _DEFAULT_CODE_JUDGE_PROMPT
CHAT_JUDGE_PRMPT = CHAT_JUDGE_PROMPT = _DEFAULT_CHAT_JUDGE_PROMPT
MULTI_TURN_JUDGE_PRMPT = MULTI_TURN_JUDGE_PROMPT = _DEFAULT_MULTI_TURN_JUDGE_PROMPT
SAC_JUDGE_PRMPT = SAC_JUDGE_PROMPT = _DEFAULT_SAC_JUDGE_PROMPT


# ============================================================================
# Orchestrator Configuration
# ============================================================================

MAX_WORKERS = 32


# ============================================================================
# Logging Configuration
# ============================================================================

def get_log_dir() -> str:
    """Get log directory from runtime config or environment."""
    return get_config("HYDRA_LOG_DIR", "./reward_logs")


def get_max_text_length() -> int:
    """Get maximum text length for logging from runtime config or environment."""
    try:
        return int(get_config("HYDRA_MAX_TEXT_LEN", "5000"))
    except ValueError:
        return 5000


def is_logging_enabled() -> bool:
    """Check if logging is enabled from runtime config or environment."""
    return get_config("HYDRA_LOGGING_ENABLED", "true").lower() in ("true", "1", "yes")
