# Refactoring Guide - HydraRP Library

## Overview

This document describes the refactoring of the HydraRP codebase from a monolithic structure to a modular, extensible library.

## Problem Statement

The original codebase consisted of three tightly-coupled files:
- `orchestrator.py` - Main orchestration logic
- `scoring_rewards.py` - Reward functions
- `llm_client.py` - LLM client implementation
- `reward_logger.py` - Logging system

Issues with the original structure:
1. **Tight coupling**: Files imported from each other creating circular dependencies
2. **Missing configuration**: Hard-coded constants and missing environment variable handling
3. **Difficult to extend**: Adding new task types or reward functions required changes across multiple files
4. **Hard to test**: No clear separation of concerns
5. **Poor reusability**: Could not easily import specific components

## Solution: Modular Library Structure

The refactored code is organized as a proper Python package under `hydra_rp/` with clear separation of concerns:

```
hydra_rp/
├── __init__.py          # Public API exports
├── config.py            # Configuration management
├── logger.py            # Logging system
├── llm_client.py        # LLM client
├── rewards.py           # Reward functions
└── orchestrator.py      # Main orchestration
```

## Key Improvements

### 1. Configuration Management (`config.py`)

**Before**: Constants scattered across files, undefined variables
```python
# In llm_client.py
judge_client = OpenAI(api_key=JUDGE_API_KEY, ...)  # JUDGE_API_KEY undefined!
```

**After**: Centralized configuration with environment variable support
```python
# In hydra_rp/config.py
def get_judge_api_key() -> str:
    return os.getenv("JUDGE_API_KEY", "")

# Usage
from hydra_rp.config import get_judge_api_key
api_key = get_judge_api_key()
```

### 2. Dependency Injection

**Before**: Direct imports creating tight coupling
```python
# scoring_rewards.py
from llm_client import call_judge_llm  # Hard dependency
```

**After**: Clean interfaces with minimal dependencies
```python
# hydra_rp/rewards.py
from .llm_client import call_judge_llm  # Relative import within package
# OR use dependency injection
from .config import get_judge_model_name  # Configuration-based
```

### 3. Clear Module Boundaries

Each module has a single, well-defined responsibility:

| Module | Responsibility |
|--------|---------------|
| `config.py` | Configuration and constants |
| `logger.py` | Logging and telemetry |
| `llm_client.py` | LLM API calls |
| `rewards.py` | Reward functions and penalties |
| `orchestrator.py` | Task routing and batch processing |

### 4. Public API (`__init__.py`)

**Before**: Users had to know internal file structure
```python
from orchestrator import legal_reward_fn
from scoring_rewards import reward_daily_chat
from llm_client import TASK_TYPE_CHAT
```

**After**: Clean, documented public API
```python
from hydra_rp import (
    legal_reward_fn,
    reward_daily_chat,
    TASK_TYPE_CHAT,
)
```

### 5. Extensibility

**Before**: Adding a new task type required editing multiple files
```python
# Had to edit orchestrator.py, scoring_rewards.py, llm_client.py
```

**After**: Simple registration system
```python
from hydra_rp import get_registry

def my_custom_reward(completion, prompt):
    return 0.8

registry = get_registry()
registry.register("custom_task", my_custom_reward)
```

## Backward Compatibility

To ensure existing code continues to work, the original files are now thin wrappers:

```python
# orchestrator.py (wrapper)
import warnings
from hydra_rp.orchestrator import legal_reward_fn

warnings.warn("Use 'from hydra_rp import ...' instead", DeprecationWarning)
```

This approach:
- ✅ Maintains 100% backward compatibility
- ✅ Provides deprecation warnings to guide migration
- ✅ Allows gradual adoption of new structure

## Migration Guide

### For End Users

**Old way:**
```python
from orchestrator import legal_reward_fn
scores = legal_reward_fn(prompts, completions, task_type=["chat"])
```

**New way (recommended):**
```python
from hydra_rp import legal_reward_fn
scores = legal_reward_fn(prompts, completions, task_type=["chat"])
```

### For Library Developers

**Adding a new reward function:**

1. Define your function in `hydra_rp/rewards.py`:
```python
def reward_new_task(completion: str, prompt: str = "") -> float:
    """Evaluate new task type."""
    # Your logic here
    return score
```

2. Add task type to `hydra_rp/config.py`:
```python
TASK_TYPE_NEW = "new_task"
```

3. Register in `hydra_rp/orchestrator.py`:
```python
reward_func_map = {
    # ... existing mappings ...
    TASK_TYPE_NEW: reward_new_task,
}
```

4. Export in `hydra_rp/__init__.py`:
```python
__all__ = [
    # ... existing exports ...
    'reward_new_task',
]
```

## Configuration

The library now supports environment variables for configuration:

```bash
# LLM Judge Configuration
export JUDGE_API_ENDPOINT="https://api.openai.com/v1/chat/completions"
export JUDGE_API_KEY="your-api-key"
export JUDGE_MODEL_NAME="gpt-4"

# Logging Configuration
export HYDRA_LOG_DIR="./reward_logs"
export HYDRA_LOGGING_ENABLED="true"
export HYDRA_MAX_TEXT_LEN="5000"
```

## Testing

The modular structure makes testing much easier:

```python
# Test a single reward function
from hydra_rp.rewards import reward_daily_chat

def test_daily_chat():
    score = reward_daily_chat("(微笑) 你好！", "你好")
    assert 0.0 <= score <= 1.0

# Test with mocked LLM client
from hydra_rp import JudgeLLMClient
from unittest.mock import Mock

client = JudgeLLMClient(api_key="test")
client.call = Mock(return_value={"content": "Score: 0.8", "reasoning_content": ""})
```

## Benefits Summary

1. **Maintainability**: Clear module boundaries make code easier to understand and modify
2. **Extensibility**: New features can be added without modifying existing code
3. **Testability**: Individual components can be tested in isolation
4. **Reusability**: Components can be imported and used independently
5. **Configuration**: Environment-based configuration for different deployments
6. **Documentation**: Comprehensive docstrings and examples
7. **Backward Compatibility**: Existing code continues to work

## File Structure Comparison

### Before
```
.
├── orchestrator.py (247 lines, mixed concerns)
├── scoring_rewards.py (584 lines, tightly coupled)
├── llm_client.py (114 lines, missing config)
└── reward_logger.py (414 lines)
```

### After
```
.
├── hydra_rp/
│   ├── __init__.py (100 lines, public API)
│   ├── config.py (180 lines, centralized config)
│   ├── logger.py (470 lines, enhanced logging)
│   ├── llm_client.py (220 lines, clean interface)
│   ├── rewards.py (750 lines, modular rewards)
│   └── orchestrator.py (420 lines, clear orchestration)
├── orchestrator.py (wrapper, 50 lines)
├── scoring_rewards.py (wrapper, 50 lines)
├── llm_client.py (wrapper, 45 lines)
├── reward_logger.py (wrapper, 35 lines)
├── README.md (comprehensive documentation)
├── example_usage.py (usage examples)
└── requirements.txt (dependencies)
```

## Next Steps

1. **Update existing code** to use new imports (optional, as wrappers provide compatibility)
2. **Add new reward functions** using the registry system
3. **Configure via environment variables** instead of hard-coding
4. **Run tests** to verify functionality
5. **Review logs** in the configured log directory

## Questions?

Refer to:
- `README.md` for usage examples
- `example_usage.py` for practical examples
- Individual module docstrings for API details
- This guide for architecture understanding
