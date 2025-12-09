"""
LLM client for judge model calls.

This module provides a clean interface for calling judge LLMs
with automatic logging and error handling.
"""

import re
import time
from typing import Dict, Any, Optional

from openai import OpenAI

from .config import (
    get_judge_api_endpoint,
    get_judge_api_key,
    get_judge_model_name,
)
from .logger import get_logger


class JudgeLLMClient:
    """Client for calling judge LLM with automatic logging."""
    
    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 2,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        timeout: float = 180.0
    ):
        """
        Initialize JudgeLLMClient.
        
        Args:
            api_endpoint: API endpoint URL
            api_key: API key for authentication
            model_name: Model name to use
            max_retries: Maximum number of retries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.api_endpoint = api_endpoint or get_judge_api_endpoint()
        self.api_key = api_key or get_judge_api_key()
        self.model_name = model_name or get_judge_model_name()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Prepare base URL
        base_url = self.api_endpoint.replace("/chat/completions", "")
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            max_retries=max_retries,
        )
    
    def call(
        self,
        system_prompt: str,
        user_content: str,
        caller_func: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call judge LLM and return response with logging.
        
        Args:
            system_prompt: System prompt
            user_content: User message content
            caller_func: Name of calling function (for logging)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
        
        Returns:
            Dict with keys:
                - reasoning_content: Reasoning/thinking process
                - content: Main response content
        """
        start_time = time.time()
        full_input = f"[System]\n{system_prompt}\n\n[User]\n{user_content}"
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temp,
                max_tokens=max_tok,
                timeout=self.timeout,
            )
            
            choice = response.choices[0]
            message = choice.message
            reasoning_content = getattr(message, "reasoning_content", "")
            
            if not reasoning_content and hasattr(message, "model_dump"):
                payload = message.model_dump()
                reasoning_content = payload.get("reasoning_content", "")
            
            result = {
                "reasoning_content": reasoning_content,
                "content": message.content or ""
            }
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger = get_logger()
            logger.log_llm_call(
                function_name=caller_func,
                model=self.model_name,
                input_prompt=full_input,
                raw_output=result["content"],
                thinking=reasoning_content,
                parsed_result=self.extract_score(result["content"]),
                latency_ms=latency_ms,
            )
            
            return result
        
        except Exception as e:
            print(f"Judge API Warning: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            logger = get_logger()
            logger.log_llm_call(
                function_name=caller_func,
                model=self.model_name,
                input_prompt=full_input,
                raw_output="",
                thinking="",
                parsed_result=0.5,
                latency_ms=latency_ms,
                error=str(e),
            )
            
            return {
                "reasoning_content": "",
                "content": f"Score: 0.5\nAnalysis: API Error ({str(e)}).",
            }
    
    @staticmethod
    def extract_score(response: str) -> float:
        """
        Extract numerical score from response text.
        
        Args:
            response: Response text containing score
        
        Returns:
            Extracted score between 0.0 and 1.0
        """
        match = re.search(r"Score:\s*([\d\.]+)", response)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0
        return 0.0


# Global client instance
_client_instance = None


def get_client() -> JudgeLLMClient:
    """Get the global JudgeLLMClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = JudgeLLMClient()
    return _client_instance


def call_judge_llm(
    system_prompt: str,
    user_content: str,
    caller_func: str = "unknown"
) -> Dict[str, Any]:
    """
    Convenience function to call judge LLM using global client.
    
    Args:
        system_prompt: System prompt
        user_content: User message content
        caller_func: Name of calling function (for logging)
    
    Returns:
        Dict with reasoning_content and content keys
    """
    client = get_client()
    return client.call(system_prompt, user_content, caller_func)


def extract_score_from_response(response: str) -> float:
    """
    Convenience function to extract score from response.
    
    Args:
        response: Response text containing score
    
    Returns:
        Extracted score between 0.0 and 1.0
    """
    return JudgeLLMClient.extract_score(response)
