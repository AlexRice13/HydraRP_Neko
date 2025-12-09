import re
from typing import Dict, Any

from openai import OpenAI
from reward_logger import reward_logger

_base_url = JUDGE_API_ENDPOINT.replace("/chat/completions", "")
if _base_url.endswith("/"):
    _base_url = _base_url[:-1]

judge_client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=_base_url,
    max_retries=2,
)

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
    if not task_type:
        return TASK_TYPE_CHAT
    return TASK_TYPE_MAPPING.get(task_type.lower().strip(), TASK_TYPE_CHAT)


def call_judge_llm(system_prompt: str, user_content: str, caller_func: str = "unknown") -> Dict[str, Any]:
    import time

    start_time = time.time()
    full_input = f"[System]\n{system_prompt}\n\n[User]\n{user_content}"

    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.6,
            max_tokens=8192,
            timeout=180.0,
        )

        choice = response.choices[0]
        message = choice.message
        reasoning_content = getattr(message, "reasoning_content", "")

        if not reasoning_content and hasattr(message, "model_dump"):
            payload = message.model_dump()
            reasoning_content = payload.get("reasoning_content", "")

        result = {"reasoning_content": reasoning_content, "content": message.content or ""}
        latency_ms = (time.time() - start_time) * 1000
        reward_logger.log_llm_call(
            function_name=caller_func,
            model=JUDGE_MODEL_NAME,
            input_prompt=full_input,
            raw_output=result["content"],
            thinking=reasoning_content,
            parsed_result=extract_score_from_response(result["content"]),
            latency_ms=latency_ms,
        )
        return result

    except Exception as e:
        print(f"Judge API Warning: {e}")
        latency_ms = (time.time() - start_time) * 1000
        reward_logger.log_llm_call(
            function_name=caller_func,
            model=JUDGE_MODEL_NAME,
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


def extract_score_from_response(response: str) -> float:
    match = re.search(r"Score:\s*([\d\.]+)", response)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0
    return 0.0