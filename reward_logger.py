# #############################################################################
# reward_logger.py - 奖励函数日志系统
# 职责：记录打分细节，包括LLM输出和推理过程，保存为CSV格式
# #############################################################################

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

TIMESTAMP = time.time()
@dataclass
class LLMCallLog:
    """单次LLM调用记录"""
    call_id: str = ""
    function_name: str = ""          # 调用来源函数
    model: str = ""                  # 模型名称
    input_prompt: str = ""           # 输入prompt
    raw_output: str = ""             # 原始输出
    thinking: str = ""               # 推理过程 (<think>标签内容)
    parsed_result: str = ""          # 解析后的结果
    latency_ms: float = 0.0          # 耗时(ms)
    error: str = ""                  # 错误信息
    timestamp: str = ""


@dataclass
class ScoringLog:
    """单次评分完整记录"""
    # 标识
    record_id: str = ""
    timestamp: str = ""
    
    # 输入
    prompt: str = ""
    completion: str = ""
    
    # 路由结果
    task_type: str = ""
    reward_function: str = ""
    
    # 评分详情
    raw_reward: float = 0.0
    penalty_repetition: float = 0.0
    penalty_length: float = 0.0
    penalty_thinking: float = 0.0
    total_penalty: float = 0.0
    final_score: float = 0.0
    
    # LLM调用列表
    llm_calls: List[LLMCallLog] = field(default_factory=list)
    
    # 额外信息
    extra: Dict[str, Any] = field(default_factory=dict)


class RewardLogger:
    """
    奖励函数日志记录器
    
    功能：
    - 记录每次评分的完整信息
    - 捕获LLM调用的输入输出和推理过程
    - 线程安全，支持并发
    - 导出为CSV格式
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        log_dir: str = "./reward_logs",
        scoring_csv: str = f"scoring_details{TIMESTAMP}.csv",
        llm_csv: str = f"llm_calls{TIMESTAMP}.csv",
        enabled: bool = True,
        max_text_len: int = 5000  # CSV中文本最大长度
    ):
        if self._initialized:
            return
        
        self.log_dir = log_dir
        self.scoring_csv_path = os.path.join(log_dir, scoring_csv)
        self.llm_csv_path = os.path.join(log_dir, llm_csv)
        self.enabled = enabled
        self.max_text_len = max_text_len
        
        self._write_lock = threading.Lock()
        self._thread_local = threading.local()
        
        # 创建目录和CSV文件
        os.makedirs(log_dir, exist_ok=True)
        self._init_csv_files()
        
        self._initialized = True
        print(f"📝 RewardLogger 已初始化 | 日志目录: {log_dir}")
    
    def _init_csv_files(self):
        """初始化CSV文件（写入表头）"""
        # 主评分日志表头
        scoring_headers = [
            "record_id", "timestamp",
            "prompt", "completion",
            "task_type", "reward_function",
            "raw_reward", "penalty_repetition", "penalty_length",
            "penalty_thinking", "total_penalty", "final_score",
            "llm_call_count",
            # 嵌入前3次LLM调用的关键信息
            "llm_0_func", "llm_0_model", "llm_0_input", "llm_0_output", 
            "llm_0_thinking", "llm_0_result", "llm_0_latency",
            "llm_1_func", "llm_1_model", "llm_1_input", "llm_1_output",
            "llm_1_thinking", "llm_1_result", "llm_1_latency",
            "llm_2_func", "llm_2_model", "llm_2_input", "llm_2_output",
            "llm_2_thinking", "llm_2_result", "llm_2_latency",
            "extra_json"
        ]
        
        # LLM调用详细日志表头
        llm_headers = [
            "call_id", "record_id", "timestamp",
            "function_name", "model",
            "input_prompt", "raw_output", "thinking",
            "parsed_result", "latency_ms", "error"
        ]
        
        # 写入表头（如果文件不存在）
        if not os.path.exists(self.scoring_csv_path):
            with open(self.scoring_csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=scoring_headers).writeheader()
        
        if not os.path.exists(self.llm_csv_path):
            with open(self.llm_csv_path, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=llm_headers).writeheader()
    
    # ========================
    # 上下文管理
    # ========================
    
    @property
    def current_log(self) -> Optional[ScoringLog]:
        """获取当前线程的评分日志"""
        return getattr(self._thread_local, 'current_log', None)
    
    @current_log.setter
    def current_log(self, log: Optional[ScoringLog]):
        """设置当前线程的评分日志"""
        self._thread_local.current_log = log
    
    @contextmanager
    def scoring_context(self, prompt: str, completion: str):
        """
        评分上下文管理器
        
        用法:
            with logger.scoring_context(prompt, completion) as log:
                log.task_type = "chat"
                log.raw_reward = 0.8
                # ... LLM调用会自动记录 ...
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
    # LLM调用记录
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
        记录一次LLM调用
        
        Args:
            function_name: 调用来源的函数名
            model: 使用的模型
            input_prompt: 发送给LLM的完整prompt
            raw_output: LLM的原始输出
            thinking: 推理过程（<think>标签内的内容）
            parsed_result: 解析后的结果（分数、JSON等）
            latency_ms: 调用耗时（毫秒）
            error: 错误信息（如果有）
        
        Returns:
            LLMCallLog: 创建的日志记录
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
        
        # 添加到当前评分日志
        if self.current_log is not None:
            self.current_log.llm_calls.append(call_log)
        
        return call_log
    
    def extract_thinking(self, text: str) -> str:
        """从文本中提取<think>标签内的推理过程"""
        if '<think>' in text and '</think>' in text:
            start = text.find('<think>') + 7
            end = text.find('</think>')
            return text[start:end].strip()
        return ""
    
    # ========================
    # 保存日志
    # ========================
    
    def _truncate(self, text: str) -> str:
        """截断过长文本"""
        if len(text) > self.max_text_len:
            return text[:self.max_text_len] + f"...[truncated, total {len(text)} chars]"
        return text
    
    def _save_log(self, log: ScoringLog):
        """保存日志到CSV"""
        with self._write_lock:
            # 1. 保存主评分日志
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
            
            # 嵌入前3次LLM调用信息
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
            
            # 写入主CSV
            with open(self.scoring_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
            
            # 2. 保存LLM调用详细日志
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
    # 工具方法
    # ========================
    
    def disable(self):
        """禁用日志"""
        self.enabled = False
    
    def enable(self):
        """启用日志"""
        self.enabled = True


# 全局单例
reward_logger = RewardLogger()


# #############################################################################
# LLM调用包装器 - 方便在奖励函数中记录LLM调用
# #############################################################################

class LLMCallWrapper:
    """
    LLM调用包装器，自动记录调用信息
    
    用法:
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
        
        # 自动提取thinking
        if not self.thinking and self.raw_output:
            self.thinking = reward_logger.extract_thinking(self.raw_output)
        
        reward_logger.log_llm_call(
            function_name=self.function_name,
            model=self.model,
            input_prompt=self.input_prompt,
            raw_output=self.raw_output,
            thinking=self.thinking,
            parsed_result=self.parsed_result,
            latency_ms=latency,
            error=self.error
        )
        
        return False  # 不抑制异常
    
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