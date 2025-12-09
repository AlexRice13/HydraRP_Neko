"""
Reward function logging system.

This module provides comprehensive logging for reward function evaluation,
including LLM call tracking and scoring details.
"""

import csv
import os
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from contextlib import contextmanager

from .config import get_log_dir, get_max_text_length, is_logging_enabled

TIMESTAMP = time.time()


@dataclass
class LLMCallLog:
    """Record of a single LLM call."""
    call_id: str = ""
    function_name: str = ""          # Calling function name
    model: str = ""                  # Model name
    input_prompt: str = ""           # Input prompt
    raw_output: str = ""             # Raw output
    thinking: str = ""               # Reasoning process (<think> tag content)
    parsed_result: str = ""          # Parsed result
    latency_ms: float = 0.0          # Latency in milliseconds
    error: str = ""                  # Error message
    timestamp: str = ""


@dataclass
class ScoringLog:
    """Complete record of a single scoring evaluation."""
    # Identifiers
    record_id: str = ""
    timestamp: str = ""
    
    # Input
    prompt: str = ""
    completion: str = ""
    
    # Routing results
    task_type: str = ""
    reward_function: str = ""
    
    # Scoring details
    raw_reward: float = 0.0
    penalty_repetition: float = 0.0
    penalty_length: float = 0.0
    penalty_thinking: float = 0.0
    total_penalty: float = 0.0
    final_score: float = 0.0
    
    # LLM call list
    llm_calls: List[LLMCallLog] = field(default_factory=list)
    
    # Extra information
    extra: Dict[str, Any] = field(default_factory=dict)


class RewardLogger:
    """
    Reward function logger.
    
    Features:
    - Records complete information for each scoring evaluation
    - Captures input/output and reasoning process of LLM calls
    - Thread-safe, supports concurrency
    - Exports to CSV format
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        scoring_csv: Optional[str] = None,
        llm_csv: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_text_len: Optional[int] = None
    ):
        if self._initialized:
            return
        
        self.log_dir = log_dir or get_log_dir()
        self.scoring_csv_path = os.path.join(
            self.log_dir, 
            scoring_csv or f"scoring_details{TIMESTAMP}.csv"
        )
        self.llm_csv_path = os.path.join(
            self.log_dir,
            llm_csv or f"llm_calls{TIMESTAMP}.csv"
        )
        self.enabled = enabled if enabled is not None else is_logging_enabled()
        self.max_text_len = max_text_len or get_max_text_length()
        
        self._write_lock = threading.Lock()
        self._thread_local = threading.local()
        
        # Create directory and CSV files
        os.makedirs(self.log_dir, exist_ok=True)
        self._init_csv_files()
        
        self._initialized = True
        # Note: Using print for initialization message as this is a one-time setup notice
        # Can be suppressed by redirecting stdout if needed
        if os.getenv("HYDRA_SILENT_INIT", "").lower() not in ("true", "1", "yes"):
            print(f"📝 RewardLogger initialized | Log directory: {self.log_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files (write headers)."""
        # Main scoring log headers
        scoring_headers = [
            "record_id", "timestamp",
            "prompt", "completion",
            "task_type", "reward_function",
            "raw_reward", "penalty_repetition", "penalty_length",
            "penalty_thinking", "total_penalty", "final_score",
            "llm_call_count",
            # Embed key information from first 3 LLM calls
            "llm_0_func", "llm_0_model", "llm_0_input", "llm_0_output", 
            "llm_0_thinking", "llm_0_result", "llm_0_latency",
            "llm_1_func", "llm_1_model", "llm_1_input", "llm_1_output",
            "llm_1_thinking", "llm_1_result", "llm_1_latency",
            "llm_2_func", "llm_2_model", "llm_2_input", "llm_2_output",
            "llm_2_thinking", "llm_2_result", "llm_2_latency",
            "extra_json"
        ]
        
        # LLM call detailed log headers
        llm_headers = [
            "call_id", "record_id", "timestamp",
            "function_name", "model",
            "input_prompt", "raw_output", "thinking",
            "parsed_result", "latency_ms", "error"
        ]
        
        # Write headers (if files don't exist)
        if not os.path.exists(self.scoring_csv_path):
            with open(self.scoring_csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=scoring_headers).writeheader()
        
        if not os.path.exists(self.llm_csv_path):
            with open(self.llm_csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=llm_headers).writeheader()
    
    # ========================
    # Context Management
    # ========================
    
    @property
    def current_log(self) -> Optional[ScoringLog]:
        """Get current thread's scoring log."""
        return getattr(self._thread_local, 'current_log', None)
    
    @current_log.setter
    def current_log(self, log: Optional[ScoringLog]):
        """Set current thread's scoring log."""
        self._thread_local.current_log = log
    
    @contextmanager
    def scoring_context(self, prompt: str, completion: str):
        """
        Scoring context manager.
        
        Usage:
            with logger.scoring_context(prompt, completion) as log:
                log.task_type = "chat"
                log.raw_reward = 0.8
                # ... LLM calls will be automatically recorded ...
        """
        log = ScoringLog(
            record_id=uuid.uuid4().hex[:8],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            prompt=prompt,
            completion=completion
        )
        self.current_log = log
        
        try:
            yield log
        finally:
            if self.enabled:
                self._save_log(log)
            self.current_log = None
    
    # ========================
    # LLM Call Recording
    # ========================
    
    def log_llm_call(
        self,
        function_name: str,
        model: str,
        input_prompt: str,
        raw_output: str,
        thinking: str = "",
        parsed_result: Any = None,
        latency_ms: float = 0.0,
        error: str = ""
    ) -> LLMCallLog:
        """
        Record an LLM call.
        
        Args:
            function_name: Name of calling function
            model: Model used
            input_prompt: Complete prompt sent to LLM
            raw_output: Raw output from LLM
            thinking: Reasoning process (<think> tag content)
            parsed_result: Parsed result (score, JSON, etc.)
            latency_ms: Call latency in milliseconds
            error: Error message if any
        
        Returns:
            LLMCallLog: Created log record
        """
        call_log = LLMCallLog(
            call_id=uuid.uuid4().hex[:8],
            function_name=function_name,
            model=model,
            input_prompt=input_prompt,
            raw_output=raw_output,
            thinking=thinking,
            parsed_result=str(parsed_result) if parsed_result is not None else "",
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        )
        
        # Add to current scoring log
        if self.current_log is not None:
            self.current_log.llm_calls.append(call_log)
        
        return call_log
    
    def extract_thinking(self, text: str) -> str:
        """Extract reasoning process from <think> tags."""
        if '<think>' in text and '</think>' in text:
            start = text.find('<think>') + 7
            end = text.find('</think>')
            return text[start:end].strip()
        return ""
    
    # ========================
    # Save Logs
    # ========================
    
    def _truncate(self, text: str) -> str:
        """Truncate overly long text."""
        if len(text) > self.max_text_len:
            return text[:self.max_text_len] + f"...[truncated, total {len(text)} chars]"
        return text
    
    def _save_log(self, log: ScoringLog):
        """Save log to CSV."""
        with self._write_lock:
            # 1. Save main scoring log
            row = {
                "record_id": log.record_id,
                "timestamp": log.timestamp,
                "prompt": self._truncate(log.prompt),
                "completion": self._truncate(log.completion),
                "task_type": log.task_type,
                "reward_function": log.reward_function,
                "raw_reward": round(log.raw_reward, 4),
                "penalty_repetition": round(log.penalty_repetition, 4),
                "penalty_length": round(log.penalty_length, 4),
                "penalty_thinking": round(log.penalty_thinking, 4),
                "total_penalty": round(log.total_penalty, 4),
                "final_score": round(log.final_score, 4),
                "llm_call_count": len(log.llm_calls),
                "extra_json": json.dumps(log.extra, ensure_ascii=False)
            }
            
            # Embed first 3 LLM call information
            for i in range(3):
                prefix = f"llm_{i}_"
                if i < len(log.llm_calls):
                    call = log.llm_calls[i]
                    row[f"{prefix}func"] = call.function_name
                    row[f"{prefix}model"] = call.model
                    row[f"{prefix}input"] = self._truncate(call.input_prompt)
                    row[f"{prefix}output"] = self._truncate(call.raw_output)
                    row[f"{prefix}thinking"] = self._truncate(call.thinking)
                    row[f"{prefix}result"] = self._truncate(call.parsed_result)
                    row[f"{prefix}latency"] = round(call.latency_ms, 2)
                else:
                    row[f"{prefix}func"] = ""
                    row[f"{prefix}model"] = ""
                    row[f"{prefix}input"] = ""
                    row[f"{prefix}output"] = ""
                    row[f"{prefix}thinking"] = ""
                    row[f"{prefix}result"] = ""
                    row[f"{prefix}latency"] = ""
            
            # Write to main CSV
            with open(self.scoring_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
            
            # 2. Save LLM call detailed logs
            if log.llm_calls:
                with open(self.llm_csv_path, 'a', newline='', encoding='utf-8') as f:
                    headers = [
                        "call_id", "record_id", "timestamp",
                        "function_name", "model",
                        "input_prompt", "raw_output", "thinking",
                        "parsed_result", "latency_ms", "error"
                    ]
                    writer = csv.DictWriter(f, fieldnames=headers)
                    
                    for call in log.llm_calls:
                        writer.writerow({
                            "call_id": call.call_id,
                            "record_id": log.record_id,
                            "timestamp": call.timestamp,
                            "function_name": call.function_name,
                            "model": call.model,
                            "input_prompt": self._truncate(call.input_prompt),
                            "raw_output": self._truncate(call.raw_output),
                            "thinking": self._truncate(call.thinking),
                            "parsed_result": self._truncate(call.parsed_result),
                            "latency_ms": round(call.latency_ms, 2),
                            "error": call.error
                        })
    
    # ========================
    # Utility Methods
    # ========================
    
    def disable(self):
        """Disable logging."""
        self.enabled = False
    
    def enable(self):
        """Enable logging."""
        self.enabled = True


# Global singleton
_logger_instance = None


def get_logger() -> RewardLogger:
    """Get the global RewardLogger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RewardLogger()
    return _logger_instance


# ============================================================================
# LLM Call Wrapper - Convenient for recording LLM calls in reward functions
# ============================================================================

class LLMCallWrapper:
    """
    LLM call wrapper that automatically records call information.
    
    Usage:
        with LLMCallWrapper("reward_chat", "gpt-4") as wrapper:
            response = your_llm_call(prompt)
            wrapper.set_output(response)
            wrapper.set_result(parsed_score)
    """
    
    def __init__(self, function_name: str, model: str, input_prompt: str = ""):
        self.function_name = function_name
        self.model = model
        self.input_prompt = input_prompt
        self.raw_output = ""
        self.thinking = ""
        self.parsed_result = None
        self.error = ""
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.time() - self._start_time) * 1000 if self._start_time else 0
        
        if exc_val:
            self.error = str(exc_val)
        
        # Auto-extract thinking
        logger = get_logger()
        if not self.thinking and self.raw_output:
            self.thinking = logger.extract_thinking(self.raw_output)
        
        logger.log_llm_call(
            function_name=self.function_name,
            model=self.model,
            input_prompt=self.input_prompt,
            raw_output=self.raw_output,
            thinking=self.thinking,
            parsed_result=self.parsed_result,
            latency_ms=latency,
            error=self.error
        )
        
        return False  # Don't suppress exceptions
    
    def set_input(self, prompt: str):
        self.input_prompt = prompt
        return self
    
    def set_output(self, output: str):
        self.raw_output = output
        return self
    
    def set_thinking(self, thinking: str):
        self.thinking = thinking
        return self
    
    def set_result(self, result: Any):
        self.parsed_result = result
        return self
