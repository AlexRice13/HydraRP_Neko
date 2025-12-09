# Using HydraRP in Jupyter Notebooks

This guide shows how to use HydraRP in Jupyter notebooks where setting environment variables is less convenient.

## Quick Start

### Cell 1: Install and Import

```python
# Install if needed
# !pip install openai

# Import the library and configuration function
from hydra_rp import set_config, legal_reward_fn
```

### Cell 2: Configure

Instead of setting environment variables, use `set_config()`:

```python
# Set all configuration in one call
set_config(
    judge_api_key="sk-your-api-key-here",
    judge_api_endpoint="https://api.openai.com/v1/chat/completions",
    judge_model_name="gpt-4",
    log_dir="./reward_logs",
    logging_enabled=True,
    max_text_len=5000
)

print("✓ Configuration set!")
```

### Cell 3: Use the Library

```python
# Prepare your data
prompts = [
    [{"role": "user", "content": "你好！今天天气真好。"}],
    [{"role": "user", "content": "计算1到100的和"}],
]

completions = [
    {"content": "(微笑) 是呀！阳光暖暖的真舒服呢~"},
    {"content": "<code_interpreter>sum(range(1, 101))</code_interpreter>"},
]

task_types = ["chat", "code"]

# Evaluate
scores = legal_reward_fn(prompts, completions, task_type=task_types)

# Display results
for i, score in enumerate(scores):
    print(f"Sample {i+1} ({task_types[i]}): {score:.3f}")
```

## Custom Judge Prompts

You can also customize the prompts used for LLM-based evaluation:

### Cell: Custom Prompts

```python
from hydra_rp import set_config

# Set custom prompts for different task types
set_config(
    judge_api_key="sk-your-key",
    
    # Custom chat evaluation prompt
    chat_judge_prompt="""
    You are evaluating a friendly chatbot response.
    
    Rate on:
    - Friendliness (40%)
    - Helpfulness (40%)
    - Brevity (20%)
    
    Output format:
    Score: [0.0-1.0]
    Analysis: [brief explanation]
    """,
    
    # Custom code evaluation prompt
    code_judge_prompt="""
    You are evaluating code quality and correctness.
    
    Rate on:
    - Correctness (50%)
    - Code style (25%)
    - Efficiency (25%)
    
    Output format:
    Score: [0.0-1.0]
    Analysis: [brief explanation]
    """,
    
    # Custom scene response prompt
    sac_judge_prompt="""
    You are evaluating roleplay quality.
    
    Rate on:
    - Character consistency (40%)
    - Immersion (30%)
    - Creativity (30%)
    
    Output format:
    Score: [0.0-1.0]
    Analysis: [brief explanation]
    """,
    
    # Custom multi-turn prompt
    multi_turn_judge_prompt="""
    You are evaluating conversation continuity.
    
    Rate on:
    - Context awareness (50%)
    - Natural flow (30%)
    - Coherence (20%)
    
    Output format:
    Score: [0.0-1.0]
    Analysis: [brief explanation]
    """
)

print("✓ Custom prompts configured!")
```

## Complete Example Workflow

```python
# Cell 1: Setup
from hydra_rp import set_config, legal_reward_fn, get_logger

set_config(
    judge_api_key="sk-...",
    judge_model_name="gpt-4",
    log_dir="./my_experiment_logs",
    logging_enabled=True
)

# Cell 2: Define evaluation data
prompts = [[{"role": "user", "content": "Hello!"}]]
completions = [{"content": "(waves) Hi there!"}]
task_types = ["chat"]

# Cell 3: Run evaluation
scores = legal_reward_fn(prompts, completions, task_type=task_types)
print(f"Score: {scores[0]:.3f}")

# Cell 4: Check logs
logger = get_logger()
print(f"Logs saved to: {logger.log_dir}")
print(f"Logging enabled: {logger.enabled}")

# Cell 5: Clear config if needed (start fresh)
from hydra_rp import clear_config
clear_config()
print("Configuration cleared!")
```

## Configuration Options

All available options for `set_config()`:

```python
set_config(
    # LLM Judge Configuration
    judge_api_endpoint="https://api.openai.com/v1/chat/completions",
    judge_api_key="sk-...",
    judge_model_name="gpt-4",
    
    # Logging Configuration
    log_dir="./reward_logs",
    logging_enabled=True,
    max_text_len=5000,
    
    # Custom Judge Prompts (optional)
    code_judge_prompt="Your custom code evaluation prompt...",
    chat_judge_prompt="Your custom chat evaluation prompt...",
    multi_turn_judge_prompt="Your custom multi-turn evaluation prompt...",
    sac_judge_prompt="Your custom scene evaluation prompt...",
)
```

## Advantages Over Environment Variables

1. **Easier in notebooks**: No need to restart kernel or use `%env`
2. **Dynamic**: Change config between cells without restarting
3. **Multiple configs**: Easy to test different configurations
4. **Clear state**: Use `clear_config()` to reset to defaults
5. **Custom prompts**: Set task-specific evaluation criteria on the fly

## Tips

- Call `set_config()` once at the beginning of your notebook
- Use `clear_config()` if you want to reset to environment variables
- Custom prompts persist until you call `clear_config()` or set new ones
- The configuration is stored in memory and doesn't affect environment variables
