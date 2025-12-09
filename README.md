# HydraRP - Reward Function Library

A comprehensive Python library for evaluating LLM model completions across multiple task types with automatic scoring, penalties, and detailed logging.

## Features

- 🎯 **Multi-Task Support**: Chat, code interpreter, image generation, scene response, multi-turn dialogue
- 🤖 **LLM-Based Judging**: Automated quality assessment using judge models
- 📊 **Comprehensive Scoring**: Raw rewards with automatic penalties for quality issues
- 📝 **Detailed Logging**: CSV-based logging of all evaluations and LLM calls
- 🔌 **Extensible Architecture**: Easy to add custom task types and reward functions
- ⚡ **Concurrent Processing**: Thread-pool based batch evaluation

## Installation

```bash
# Install required dependencies
pip install openai
```

## Quick Start

### Basic Usage

```python
from hydra_rp import legal_reward_fn

# Prepare data
prompts = [[{"role": "user", "content": "你好！"}]]
completions = [{"content": "(微笑) 你好！很高兴见到你。"}]

# Evaluate
scores = legal_reward_fn(
    prompts=prompts,
    completions=completions,
    task_type=["chat"]
)

print(f"Score: {scores[0]}")
```

### Configuration

#### Option 1: Environment Variables (Traditional)

```bash
export JUDGE_API_ENDPOINT="https://api.openai.com/v1/chat/completions"
export JUDGE_API_KEY="your-api-key"
export JUDGE_MODEL_NAME="gpt-4"
export HYDRA_LOG_DIR="./reward_logs"
export HYDRA_LOGGING_ENABLED="true"
```

#### Option 2: Programmatic Configuration (Jupyter Notebooks)

Perfect for Jupyter notebooks where environment variables are less convenient:

```python
from hydra_rp import set_config

set_config(
    judge_api_key="your-api-key",
    judge_model_name="gpt-4",
    log_dir="./reward_logs",
    logging_enabled=True,
    # Optional: customize evaluation prompts
    chat_judge_prompt="Your custom evaluation prompt...",
    code_judge_prompt="Your custom code evaluation prompt...",
)
```

See [JUPYTER_USAGE.md](JUPYTER_USAGE.md) for detailed Jupyter notebook examples.

## Task Types

The library supports the following task types:

- **`daily_chat`**: Regular conversational responses
- **`code_interpreter`**: Code generation and execution tasks
- **`image_gen`**: Image generation prompt creation
- **`scene_response`**: Scenario-based roleplay responses
- **`multi_turn_chat`**: Multi-turn dialogue with context

## Architecture

### Module Structure

```
hydra_rp/
├── __init__.py          # Public API exports
├── config.py            # Configuration and constants
├── logger.py            # Logging system
├── llm_client.py        # LLM judge client
├── rewards.py           # Reward functions and penalties
└── orchestrator.py      # Main orchestration logic
```

### Key Components

1. **Orchestrator** (`orchestrator.py`): Main entry point, handles batch processing and task routing
2. **Reward Functions** (`rewards.py`): Specialized evaluation logic for each task type
3. **LLM Client** (`llm_client.py`): Interface for calling judge models
4. **Logger** (`logger.py`): Comprehensive logging of evaluations and LLM calls
5. **Config** (`config.py`): Centralized configuration management

## Advanced Usage

### Custom Reward Functions

```python
from hydra_rp import get_registry

def my_custom_reward(completion: str, prompt: str = "") -> float:
    """Custom reward function."""
    # Your scoring logic here
    score = 0.8
    return score

# Register custom task type
registry = get_registry()
registry.register("custom_task", my_custom_reward)

# Use it
scores = legal_reward_fn(
    prompts=prompts,
    completions=completions,
    task_type=["custom_task"]
)
```

### Direct Reward Function Usage

```python
from hydra_rp import reward_daily_chat, reward_code_interpreter

# Use reward functions directly
score = reward_daily_chat(
    completion="(微笑) 你好！",
    prompt="你好"
)

code_score = reward_code_interpreter(
    completion="<code_interpreter>print('Hello')</code_interpreter>",
    prompt="打印Hello"
)
```

### Custom LLM Client

```python
from hydra_rp import JudgeLLMClient

# Create custom client
client = JudgeLLMClient(
    api_endpoint="https://custom-api.com/v1/chat/completions",
    api_key="your-key",
    model_name="custom-model",
    temperature=0.7
)

# Use it
response = client.call(
    system_prompt="You are a helpful judge.",
    user_content="Evaluate this response...",
    caller_func="my_function"
)
```

### Logging

Logs are automatically saved to CSV files:

- `scoring_details_{timestamp}.csv`: Complete scoring information
- `llm_calls_{timestamp}.csv`: Detailed LLM call logs

```python
from hydra_rp import get_logger

logger = get_logger()

# Disable logging
logger.disable()

# Re-enable logging
logger.enable()

# Use logging context manually
with logger.scoring_context(prompt, completion) as log:
    log.task_type = "chat"
    log.raw_reward = 0.8
    # LLM calls within this context are automatically logged
```

## Scoring System

### Components

1. **Raw Reward**: Task-specific base score (0.0 - 1.0)
2. **Penalties**:
   - Repetition penalty: Detects repetitive content
   - Length penalty: Checks if response is too short/long
   - Thinking format penalty: Validates `<think>` tag usage

### Final Score

```
final_score = raw_reward - (penalty_repetition + penalty_length + penalty_thinking)
```

## Examples

### Example 1: Chat Evaluation

```python
from hydra_rp import legal_reward_fn

prompts = [[
    {"role": "user", "content": "今天天气真好！"}
]]
completions = [{
    "content": "<think>用户在分享好天气，应该积极回应</think>(开心地摇晃尾巴) 是呀！阳光暖暖的真舒服呢~"
}]

scores = legal_reward_fn(prompts, completions, task_type=["chat"])
print(f"Chat score: {scores[0]:.3f}")
```

### Example 2: Code Evaluation

```python
prompts = [[
    {"role": "user", "content": "计算1到100的和"}
]]
completions = [{
    "content": "<code_interpreter>result = sum(range(1, 101))\nprint(result)</code_interpreter>"
}]

scores = legal_reward_fn(prompts, completions, task_type=["code"])
print(f"Code score: {scores[0]:.3f}")
```

### Example 3: Batch Evaluation

```python
# Multiple samples at once
prompts = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "写一个斐波那契函数"}],
    [{"role": "user", "content": "给我画一只猫"}],
]

completions = [
    {"content": "(微笑) Hi there!"},
    {"content": "<code_interpreter>def fib(n): ...</code_interpreter>"},
    {"content": '{"prompt": "cute cat, anime style"}'},
]

task_types = ["chat", "code", "image_gen"]

scores = legal_reward_fn(
    prompts=prompts,
    completions=completions,
    task_type=task_types
)

for i, score in enumerate(scores):
    print(f"Sample {i+1} ({task_types[i]}): {score:.3f}")
```

## Migration from Legacy Code

If you're migrating from the old monolithic files:

### Old Code

```python
from orchestrator import legal_reward_fn
from scoring_rewards import reward_daily_chat
from llm_client import call_judge_llm
from reward_logger import reward_logger
```

### New Code

```python
from hydra_rp import (
    legal_reward_fn,
    reward_daily_chat,
    call_judge_llm,
    get_logger
)

# Get logger instance
logger = get_logger()
```

All functionality is preserved, but now with better organization and extensibility!

## Contributing

The library is designed to be extended. To add a new task type:

1. Define your reward function in `rewards.py`
2. Add task type constant to `config.py`
3. Register it in the orchestrator or use the registry
4. Update the public API in `__init__.py`

## License

This library is provided as-is for research and development purposes.
