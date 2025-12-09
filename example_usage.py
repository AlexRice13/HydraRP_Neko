#!/usr/bin/env python3
"""
Example usage of the HydraRP library.

This script demonstrates various ways to use the library for
evaluating model completions.
"""

import os

# Set up environment variables (if not already set)
os.environ.setdefault("JUDGE_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")
os.environ.setdefault("JUDGE_API_KEY", "your-api-key-here")
os.environ.setdefault("JUDGE_MODEL_NAME", "gpt-4")
os.environ.setdefault("HYDRA_LOGGING_ENABLED", "true")
os.environ.setdefault("HYDRA_LOG_DIR", "./example_logs")

from hydra_rp import (
    legal_reward_fn,
    reward_daily_chat,
    reward_code_interpreter,
    get_registry,
    get_logger,
)


def example_basic_usage():
    """Example 1: Basic usage with legal_reward_fn."""
    print("=" * 60)
    print("Example 1: Basic Chat Evaluation")
    print("=" * 60)
    
    prompts = [[{"role": "user", "content": "你好！今天天气真好。"}]]
    completions = [{
        "content": "<think>用户在打招呼并分享好天气，我应该积极回应</think>(开心地摇晃尾巴) 是呀！阳光暖暖的真舒服呢~ (眯起眼睛享受阳光)"
    }]
    
    scores = legal_reward_fn(
        prompts=prompts,
        completions=completions,
        task_type=["chat"]
    )
    
    print(f"Score: {scores[0]:.3f}")
    print()


def example_direct_reward_function():
    """Example 2: Using reward functions directly."""
    print("=" * 60)
    print("Example 2: Direct Reward Function Usage")
    print("=" * 60)
    
    completion = "(微笑) 你好！很高兴见到你。"
    prompt = "你好"
    
    score = reward_daily_chat(completion, prompt)
    print(f"Chat score: {score:.3f}")
    print()


def example_code_evaluation():
    """Example 3: Code interpreter evaluation."""
    print("=" * 60)
    print("Example 3: Code Evaluation")
    print("=" * 60)
    
    prompts = [[{"role": "user", "content": "计算1到100的和"}]]
    completions = [{
        "content": "<code_interpreter>result = sum(range(1, 101))\nprint(result)</code_interpreter>\n\n计算结果是5050。"
    }]
    
    scores = legal_reward_fn(
        prompts=prompts,
        completions=completions,
        task_type=["code"]
    )
    
    print(f"Code score: {scores[0]:.3f}")
    print()


def example_batch_evaluation():
    """Example 4: Batch evaluation with multiple samples."""
    print("=" * 60)
    print("Example 4: Batch Evaluation")
    print("=" * 60)
    
    prompts = [
        [{"role": "user", "content": "你好"}],
        [{"role": "user", "content": "写一个计算斐波那契数列的函数"}],
        [{"role": "user", "content": "帮我生成一张可爱猫娘的图片"}],
    ]
    
    completions = [
        {"content": "(微笑) 你好呀！"},
        {"content": "<code_interpreter>def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n\nprint(fib(10))</code_interpreter>"},
        {"content": '```json\n{"prompt": "cute anime cat girl, green eyes, blonde hair, orange cat ears, smiling, high quality illustration"}\n```'},
    ]
    
    task_types = ["chat", "code", "image_gen"]
    
    scores = legal_reward_fn(
        prompts=prompts,
        completions=completions,
        task_type=task_types
    )
    
    for i, score in enumerate(scores):
        print(f"Sample {i+1} ({task_types[i]}): {score:.3f}")
    print()


def example_custom_reward_function():
    """Example 5: Registering and using custom reward function."""
    print("=" * 60)
    print("Example 5: Custom Reward Function")
    print("=" * 60)
    
    def custom_reward(completion: str, prompt: str = "") -> float:
        """Simple custom reward based on length."""
        length = len(completion)
        if 50 <= length <= 200:
            return 1.0
        elif length < 50:
            return 0.5
        else:
            return 0.7
    
    # Register custom task type
    registry = get_registry()
    registry.register("custom_length_check", custom_reward)
    
    prompts = [[{"role": "user", "content": "Test"}]]
    completions = [{"content": "This is a test response with reasonable length."}]
    
    scores = legal_reward_fn(
        prompts=prompts,
        completions=completions,
        task_type=["custom_length_check"]
    )
    
    print(f"Custom task score: {scores[0]:.3f}")
    print(f"Available task types: {registry.list_task_types()}")
    print()


def example_logger_control():
    """Example 6: Controlling the logger."""
    print("=" * 60)
    print("Example 6: Logger Control")
    print("=" * 60)
    
    logger = get_logger()
    
    print(f"Logger enabled: {logger.enabled}")
    print(f"Log directory: {logger.log_dir}")
    
    # Temporarily disable logging
    logger.disable()
    print("Logging disabled")
    
    # Run evaluation without logging
    prompts = [[{"role": "user", "content": "Test"}]]
    completions = [{"content": "Test response"}]
    scores = legal_reward_fn(prompts, completions, task_type=["chat"])
    
    # Re-enable logging
    logger.enable()
    print("Logging re-enabled")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HydraRP Library Usage Examples")
    print("=" * 60 + "\n")
    
    # Note: Most examples require a valid API key to run fully
    print("Note: These examples demonstrate the API usage.")
    print("To run with actual evaluation, set JUDGE_API_KEY environment variable.\n")
    
    example_basic_usage()
    example_direct_reward_function()
    example_code_evaluation()
    example_batch_evaluation()
    example_custom_reward_function()
    example_logger_control()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
