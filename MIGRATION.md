# Migration Guide

## For Users: Migrating to the New API

The original files (`orchestrator.py`, `scoring_rewards.py`, `llm_client.py`, `reward_logger.py`) continue to work but show deprecation warnings. Here's how to migrate:

### Before (Old Way - Still Works)

```python
from orchestrator import legal_reward_fn
from scoring_rewards import reward_daily_chat
from llm_client import TASK_TYPE_CHAT
from reward_logger import reward_logger

# Use functions
scores = legal_reward_fn(prompts, completions, task_type=["chat"])
```

### After (New Way - Recommended)

```python
from hydra_rp import (
    legal_reward_fn,
    reward_daily_chat,
    TASK_TYPE_CHAT,
    get_logger
)

# Use functions (same API)
scores = legal_reward_fn(prompts, completions, task_type=["chat"])

# Get logger instance
logger = get_logger()
```

### Migration Checklist

- [ ] Replace `from orchestrator import ...` with `from hydra_rp import ...`
- [ ] Replace `from scoring_rewards import ...` with `from hydra_rp import ...`
- [ ] Replace `from llm_client import ...` with `from hydra_rp import ...`
- [ ] Replace `from reward_logger import reward_logger` with `from hydra_rp import get_logger`
- [ ] Set environment variables for configuration (see below)
- [ ] Test your code with the new imports

### Environment Variables

Configure the library using environment variables:

```bash
# Required
export JUDGE_API_KEY="your-api-key"

# Optional (with defaults)
export JUDGE_API_ENDPOINT="https://api.openai.com/v1/chat/completions"
export JUDGE_MODEL_NAME="gpt-4"
export HYDRA_LOG_DIR="./reward_logs"
export HYDRA_LOGGING_ENABLED="true"
export HYDRA_MAX_TEXT_LEN="5000"
export HYDRA_SILENT_INIT="false"
```

## Benefits of Migration

1. **Cleaner imports** - All from one package
2. **Better configuration** - Environment variables instead of hard-coded values
3. **Extensibility** - Easy to add custom task types
4. **Better organization** - Clear module boundaries
5. **Future-proof** - Old files may be removed in future versions

## No Breaking Changes

The new library maintains 100% API compatibility:
- All function signatures are the same
- All return values are the same
- All behavior is the same
- Only the import paths have changed

## Need Help?

- Check `README.md` for usage examples
- Check `REFACTORING_GUIDE.md` for architecture details
- Run `example_usage.py` for working examples
