"""
Configuration module for HydraRP library.

This module contains all configuration constants and environment variables
used throughout the library.
"""

import os
from typing import Optional

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
    """Get judge API endpoint from environment."""
    return os.getenv("JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")


def get_judge_api_key() -> str:
    """Get judge API key from environment."""
    return os.getenv("JUDGE_API_KEY", "")


def get_judge_model_name() -> str:
    """Get judge model name from environment."""
    return os.getenv("JUDGE_MODEL_NAME", "gpt-4")


# ============================================================================
# Judge Prompts
# ============================================================================

CODE_JUDGE_PROMPT = """You are a code execution and correctness evaluator.

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

CHAT_JUDGE_PROMPT = """You are evaluating a chat response quality.

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

MULTI_TURN_JUDGE_PROMPT = """You are evaluating multi-turn dialogue quality.

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

SAC_JUDGE_PROMPT = """You are evaluating a scene/scenario response.

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


# ============================================================================
# Orchestrator Configuration
# ============================================================================

MAX_WORKERS = 32


# ============================================================================
# Logging Configuration
# ============================================================================

def get_log_dir() -> str:
    """Get log directory from environment."""
    return os.getenv("HYDRA_LOG_DIR", "./reward_logs")


def get_max_text_length() -> int:
    """Get maximum text length for logging."""
    try:
        return int(os.getenv("HYDRA_MAX_TEXT_LEN", "5000"))
    except ValueError:
        return 5000


def is_logging_enabled() -> bool:
    """Check if logging is enabled."""
    return os.getenv("HYDRA_LOGGING_ENABLED", "true").lower() in ("true", "1", "yes")
