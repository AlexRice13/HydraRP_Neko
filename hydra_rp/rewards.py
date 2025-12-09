"""
Reward functions for different task types.

This module contains specialized reward functions for evaluating
model completions across different task categories.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Tuple, List

from .config import (
    TASK_TYPE_CHAT,
    TASK_TYPE_CODE,
    TASK_TYPE_IMAGE,
    TASK_TYPE_MULTI,
    TASK_TYPE_SCENE,
    get_code_judge_prompt,
    get_chat_judge_prompt,
    get_multi_turn_judge_prompt,
    get_sac_judge_prompt,
)
from .llm_client import call_judge_llm, extract_score_from_response


# ============================================================================
# Character Set for Gibberish Detection
# ============================================================================

KAOMOJI_CHARS = set(
    "◕◔◉◎●○◐◑◒◓◦◯"
    "ᴗωーへ∀▽△▼▲∇□■◇◆"
    "╥╯╰つっノシ彡ミ丿乀"
    "゜゛°•·・♥♡♦♢♠♣★☆✧✦✩✪✫✬✭✮✯❤❥❣"
    "＾^ˇˆ`´¨~∼≈"
    "εз3ε"
    "｡。，、！？…—"
    "（）()「」【】『』《》〈〉"
    "♪♫♬♩🎵"
    "σД;ﾟдTQAOo0Зз"
    "இ☉Θθ"
    "ノヽヾ〃々仝ゞゝ"
    "←→↑↓↔↕"
    "⊃⊂∩∪⊆⊇"
    "≧≦＞＜><"
    "ーァィゥェォャュョッンー"
)


# ============================================================================
# Code Execution Utilities
# ============================================================================

def execute_code_safely(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code safely in a subprocess.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
    
    Returns:
        Dict with status, output, and return_code
    """
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            lines = code.strip().splitlines()
            if lines:
                last_line = lines[-1]
                if not last_line.strip().startswith("print("):
                    code_lines = lines[:-1]
                    code_lines.append(f"print({last_line})")
                    code = "\n".join(code_lines)
    except Exception:
        pass

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_file = f.name

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        output = result.stdout.strip()
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr.strip()}"

        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": output if output else "(无输出结果)",
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "output": f"超时 ({timeout}s)",
            "return_code": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "output": f"系统异常: {str(e)}",
            "return_code": -1
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


# ============================================================================
# Reward Functions
# ============================================================================

def reward_code_interpreter(completion: str, prompt: str = "") -> float:
    """
    Evaluate code interpreter task completion.
    
    Args:
        completion: Model's completion text
        prompt: Original user prompt
    
    Returns:
        Score between 0.0 and 1.0
    """
    xml_pattern = r"""
        <code_interpreter\s+
        [^>]*
        >
        (.*?)
        </code_interpreter>
    """
    match = re.search(xml_pattern, completion, re.VERBOSE | re.DOTALL | re.IGNORECASE)
    if not match:
        return 0.1

    raw_code = match.group(1).strip()
    format_multiplier = 1.0

    if "```" in raw_code:
        format_multiplier = 0.4
        clean_code = re.sub(r"^```[a-zA-Z]*\n", "", raw_code)
        clean_code = re.sub(r"\n```$", "", clean_code)
        clean_code = clean_code.replace("```", "")
        code_content = clean_code.strip()
    else:
        code_content = raw_code

    if not code_content:
        return 0.0

    exec_result = execute_code_safely(code_content)

    if exec_result["status"] == "success":
        run_score = 0.3 if exec_result["output"] != "(无输出结果)" else 0.15
    elif exec_result["status"] == "timeout":
        run_score = 0.0
    else:
        run_score = 0.1

    judge_system_prompt = get_code_judge_prompt()
    judge_input = (
        f"问题: {prompt}\n"
        f"模型原始输出：{completion}\n"
        f"代码: {code_content}\n"
        f"运行输出: {exec_result['output']}"
    )

    try:
        judge_res = call_judge_llm(
            judge_system_prompt, judge_input, "reward_code_interpreter"
        )
        solve_score = extract_score_from_response(judge_res["content"]) * 0.7
    except Exception:
        solve_score = 0.0

    final_score = format_multiplier * (run_score + solve_score)
    return max(min(final_score, 1.0), 0.0) ** 2


def reward_image_gen(completion: str, prompt: str = "") -> float:
    """
    Evaluate image generation prompt quality.
    
    Args:
        completion: Model's completion text
        prompt: Original user prompt
    
    Returns:
        Score between 0.0 and 1.0
    """
    target_self_re = re.compile(
        r"(自拍|自画像|你自己|selfie|portrait|picture of you|photo of you|你|玲奈|asahi|lina|me\b)",
        re.IGNORECASE
    )
    trait_species = re.compile(
        r"(cat\s*girl|neko|feline|ear|tail|猫娘|猫耳|兽耳|尾巴|猫咪)",
        re.IGNORECASE
    )
    trait_eyes = re.compile(r"(green|emerald|agate|绿|翠|玛瑙)", re.IGNORECASE)
    trait_hair = re.compile(r"(blonde|gold|hair|金|发)", re.IGNORECASE)
    trait_ear_color = re.compile(
        r"(orange|ginger|tangerine|yellow|橘|橙|黄)",
        re.IGNORECASE
    )
    trait_body = re.compile(
        r"(petite|small|short|155|girl|young|lady|teen|少女|萝莉|女孩|娇小|个子)",
        re.IGNORECASE
    )
    style_keywords = re.compile(
        r"(anime|manga|comic|illustration|realistic|photo|quality|light|detail|art|style|动漫|二次元|插画|绘画|写实|摄影|画质|光影|风格|作画)",
        re.IGNORECASE
    )

    score = 0.0

    if "```" in completion:
        score += 0.05

    json_match = re.search(r"\{.*\}", completion, re.DOTALL)
    if not json_match:
        return 0.0

    json_str = json_match.group(0)
    prompt_text = ""

    try:
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            sanitized = (
                json_str.replace("\\n", " ")
                .replace('\\"', '"')
                .replace("\n", " ")
            )
            data = json.loads(sanitized)

        if "prompt" in data and isinstance(data.get("prompt"), str):
            prompt_text = data["prompt"].strip()
            score += 0.15
        else:
            return 0.1
    except Exception:
        return 0.1

    if not prompt_text:
        return score

    token_proxy_len = (
        len(prompt_text)
        if any("\u4e00" <= c <= "\u9fff" for c in prompt_text)
        else len(prompt_text.split())
    )
    length_score = min(0.15, max(0.0, token_proxy_len / 100.0 * 0.15))
    score += length_score

    style_matches = len(style_keywords.findall(prompt_text))
    style_score = min(0.15, style_matches * 0.05)
    score += style_score

    is_self_request = target_self_re.search(prompt) is not None

    if is_self_request:
        traits_detected = 0
        traits_total = 5.0
        if trait_species.search(prompt_text):
            traits_detected += 1
        if trait_eyes.search(prompt_text):
            traits_detected += 1
        # Hair trait already includes gold/金, just check for additional blonde/yellow keywords
        if trait_hair.search(prompt_text):
            traits_detected += 1
        if trait_ear_color.search(prompt_text):
            traits_detected += 1
        if trait_body.search(prompt_text):
            traits_detected += 1
        consistency_score = (traits_detected / traits_total) * 0.5
        score += consistency_score
    else:
        extra_detail_score = min(0.5, token_proxy_len / 150.0 * 0.5)
        score += extra_detail_score

    return max(0.0, min(score, 1.0))


def reward_daily_chat(completion: str, prompt: str = "") -> float:
    """
    Evaluate daily chat response quality.
    
    Args:
        completion: Model's completion text
        prompt: Original user prompt
    
    Returns:
        Score between 0.0 and 1.0
    """
    if re.search(r"\(.*?\)", completion):
        rule_score = 1.0
    else:
        rule_score = 0.0

    try:
        judge_res = call_judge_llm(
            get_chat_judge_prompt(),
            f"User: {prompt}\nResponse: {completion}",
            caller_func="reward_daily_chat",
        )
        llm_raw_score = extract_score_from_response(judge_res["content"])
    except Exception:
        llm_raw_score = 0.0

    if llm_raw_score < 0.8:
        llm_transformed = llm_raw_score
    else:
        llm_transformed = llm_raw_score + (llm_raw_score - 0.8) ** 2

    final_score = (rule_score * 0.2) + (llm_transformed * 0.8)
    return max(0.0, min(final_score, 1.0))


def reward_multi_turn_chat(completion: str, prompt: str = "") -> float:
    """
    Evaluate multi-turn chat response quality.
    
    Args:
        completion: Model's completion text
        prompt: Dialogue history in ChatML format
    
    Returns:
        Score between 0.0 and 1.0
    """
    match_action = re.search(r"\(.*?\)|（.*?）", completion)
    rule_score = 1.0 if match_action else 0.0

    try:
        judge_input = (
            f"[Dialogue History (ChatML)]:\n{prompt}\n\n"
            f"[Current Response Analysis Target]:\n{completion}"
        )
        judge_res = call_judge_llm(
            system_prompt=get_multi_turn_judge_prompt(),
            user_content=judge_input,
            caller_func="reward_multi_turn_chat",
        )
        llm_raw_score = extract_score_from_response(judge_res["content"])
    except Exception as e:
        print(f"Multi-turn Judge Error: {e}")
        llm_raw_score = 0.0

    final_score = (rule_score * 0.15) + (llm_raw_score * 0.85)
    transformed_score = max(0.0, min(final_score, 1.0)) ** 2.5
    return transformed_score


def reward_scene_response(completion: str, prompt: str = "") -> float:
    """
    Evaluate scene/scenario response quality.
    
    Args:
        completion: Model's completion text
        prompt: Scene/scenario description
    
    Returns:
        Score between 0.0 and 1.0
    """
    think_content, response = extract_think_and_response(completion)
    score = 0.4

    gibberish_ratio, abnormal_chars = detect_gibberish_chars(completion)
    if gibberish_ratio > 0.02:
        return 0.05
    elif gibberish_ratio > 0.005:
        score -= 0.15

    thought_issues = detect_thought_collapse(think_content)
    if thought_issues["meta_leak"]:
        return 0.25
    if thought_issues["empty_think"]:
        score -= 0.2
    if thought_issues["repetition"]:
        score -= 0.15
    if thought_issues["persona_break"]:
        score -= 0.1
    if thought_issues["fragmented"]:
        score -= 0.05

    response_issues = detect_response_quality(response)
    if response_issues["format_chaos"]:
        score -= 0.1
    if response_issues["truncated"]:
        score -= 0.1
    if response_issues["excessive_kaomoji"]:
        score -= 0.05

    if not re.search(r"[（(].+?[）)]", completion):
        score -= 0.1
    if "```" in completion:
        score -= 0.1

    if score < 0.15:
        return max(0.05, score)

    judge_res = call_judge_llm(
        get_sac_judge_prompt(),
        f"Prompt: {prompt}\nResponse: {completion}",
        caller_func="reward_scene_response",
    )
    llm_score = extract_score_from_response(judge_res["content"])
    effective_llm_weight = 0.5 * (score / 0.4)
    score = score + llm_score * effective_llm_weight

    final_score = max(0.0, min(score, 1.0))
    return final_score


# ============================================================================
# Penalty Functions
# ============================================================================

def calculate_repetition_penalty(completion: str, task_type: str = "") -> float:
    """
    Calculate penalty for repetitive content.
    
    Args:
        completion: Model's completion text
        task_type: Task type identifier
    
    Returns:
        Penalty score between 0.0 and 1.0
    """
    if not completion or len(completion) < 10:
        return 0.0

    raw_penalty_score = 0.0

    def find_repeated_patterns(text: str, min_repeats: int = 2):
        chunk_len = max(10, min(len(text) // 5, 50))
        patterns = set()
        seen_chunks = set()
        step = max(1, chunk_len // 2)

        for i in range(0, len(text) - chunk_len + 1, step):
            substr = text[i : i + chunk_len]
            if not substr.strip():
                continue
            if re.match(r"^[\s\W_]+$", substr):
                continue  # Skip pure whitespace/punctuation patterns
            if substr in seen_chunks:
                continue
            seen_chunks.add(substr)
            if text.count(substr) >= min_repeats:
                patterns.add(substr)
        return patterns

    repeated_substrings = find_repeated_patterns(completion)
    if repeated_substrings:
        raw_penalty_score += len(repeated_substrings) * 0.15

    normalized_text = re.sub(r"\n+", " ", completion)
    sentences = re.split(r"[。！？.!?]+", normalized_text)

    sentence_freq = {}
    for s in sentences:
        s = s.strip()
        if len(s) > 8:
            sentence_freq[s] = sentence_freq.get(s, 0) + 1

    max_sent_repeat = 0
    for count in sentence_freq.values():
        if count > max_sent_repeat:
            max_sent_repeat = count

    if max_sent_repeat >= 2:
        raw_penalty_score += 0.2 * (max_sent_repeat - 1)

    words = re.findall(r"\b[a-zA-Z\u4e00-\u9fa5]+\b", completion)
    if len(words) > 20:
        from collections import Counter

        counts = Counter(words)
        top_word, top_count = counts.most_common(1)[0]
        if len(top_word) > 1:
            ratio = top_count / len(words)
            if ratio > 0.35:
                raw_penalty_score += 0.3 + (ratio - 0.35) * 2.0

    if task_type == TASK_TYPE_CHAT:
        raw_penalty_score *= 0.8

    final_penalty = (raw_penalty_score * 1.2) ** 2
    return min(final_penalty, 1.0)


def calculate_length_penalty(completion: str, task_type: str = "") -> float:
    """
    Calculate penalty for inappropriate response length.
    
    Args:
        completion: Model's completion text
        task_type: Task type identifier
    
    Returns:
        Penalty score between 0.0 and 1.0
    """
    if not completion:
        return 0.0

    length = len(completion)
    length_configs = {
        TASK_TYPE_CODE: (100, 768, 1960, 2048),
        TASK_TYPE_IMAGE: (80, 486, 512, 1024),
        TASK_TYPE_CHAT: (80, 486, 617, 1024),
        TASK_TYPE_SCENE: (80, 486, 617, 1024),
        TASK_TYPE_MULTI: (150, 512, 1024, 1536),
        "default": (20, 40, 200, 500),
    }

    min_len, opt_min_len, opt_max_len, max_len = length_configs.get(
        task_type, length_configs["default"]
    )

    if opt_min_len <= length <= opt_max_len:
        return 0.0

    if length < min_len:
        return 0.2
    elif length < opt_min_len:
        ratio = (opt_min_len - length) / (opt_min_len - min_len)
        return ratio * 0.1
    elif length > max_len:
        return 1.0
    elif length > opt_max_len:
        ratio = (length - opt_max_len) / (max_len - opt_max_len)
        return min(1.0, (ratio**2) + 0.1)

    return 0.0


def calculate_thinking_format_penalty(completion: str, task_type: str = "") -> float:
    """
    Calculate penalty for improper thinking format.
    
    Args:
        completion: Model's completion text
        task_type: Task type identifier
    
    Returns:
        Penalty score between 0.0 and 1.0
    """
    if not completion:
        return 1.0

    weights = {
        "format_integrity": 0.5,
        "length_compliance": 0.5,
    }

    think_length_configs = {
        TASK_TYPE_CODE: (50, 420, 768, 1024),
        TASK_TYPE_IMAGE: (20, 200, 384, 512),
        TASK_TYPE_CHAT: (20, 300, 384, 512),
        TASK_TYPE_SCENE: (50, 300, 384, 512),
        TASK_TYPE_MULTI: (50, 350, 450, 800),
        "default": (10, 30, 200, 300),
    }

    min_len, opt_min_len, opt_max_len, max_len = think_length_configs.get(
        task_type, think_length_configs["default"]
    )

    penalty_score = 0.0
    think_tag_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_tag_pattern, completion, re.DOTALL | re.IGNORECASE)

    if not matches:
        return 1.0

    thinking_content = matches[0].strip()
    match_obj = re.search(think_tag_pattern, completion, re.IGNORECASE)
    if match_obj and match_obj.start() > 50:
        penalty_score += weights["format_integrity"] * 0.8
    elif len(matches) > 1:
        penalty_score += weights["format_integrity"] * 0.5

    think_len = len(thinking_content)
    length_penalty = 0.0

    if think_len == 0:
        length_penalty = 1.0
    elif opt_min_len <= think_len <= opt_max_len:
        length_penalty = 0.0
    elif think_len < opt_min_len:
        if think_len < min_len:
            length_penalty = 0.5
        else:
            ratio = (think_len - min_len) / (opt_min_len - min_len)
            length_penalty = 0.2 * (1.0 - ratio)
    elif think_len > opt_max_len:
        if think_len >= max_len:
            length_penalty = 1.0
        else:
            ratio = (think_len - opt_max_len) / (max_len - opt_max_len)
            length_penalty = ratio**1.5

    penalty_score += weights["length_compliance"] * length_penalty
    final_penalty = max(0.0, min(1.0, penalty_score))
    return final_penalty


# ============================================================================
# Quality Detection Utilities
# ============================================================================

def is_allowed_char(char: str) -> bool:
    """Check if a character is allowed (not gibberish)."""
    if "\x20" <= char <= "\x7e":
        return True
    if "\u4e00" <= char <= "\u9fff":
        return True
    if "\u3040" <= char <= "\u30ff":
        return True
    if "\uff00" <= char <= "\uffef":
        return True
    if char in KAOMOJI_CHARS:
        return True
    if "\u3000" <= char <= "\u303f":
        return True
    if "\U0001F300" <= char <= "\U0001F9FF":
        return True
    return False


def detect_gibberish_chars(text: str) -> Tuple[float, List[str]]:
    """Detect gibberish characters in text."""
    if not text:
        return 0.0, []

    abnormal_chars = []
    for char in text:
        if char in "\n\r\t":
            continue
        if not is_allowed_char(char):
            abnormal_chars.append(char)

    ratio = len(abnormal_chars) / len(text) if text else 0
    return ratio, list(set(abnormal_chars))


def detect_thought_collapse(think_content: str) -> Dict:
    """Detect issues in thinking content."""
    issues = {
        "repetition": False,
        "fragmented": False,
        "meta_leak": False,
        "persona_break": False,
        "empty_think": False,
    }

    if not think_content or len(think_content.strip()) < 10:
        issues["empty_think"] = True
        return issues

    if re.search(r"(.{3,30})\1{2,}", think_content):
        issues["repetition"] = True

    sentences = re.split(r"[。！？\n]", think_content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) >= 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            issues["repetition"] = True

    fragment_markers = len(re.findall(r"\.{3,}|…{2,}|—{2,}|－{2,}", think_content))
    if fragment_markers > 8:
        issues["fragmented"] = True

    meta_patterns = [
        r"作为(?:一个)?(?:AI|语言模型|助手)",
        r"我(?:是|在)(?:扮演|模拟|roleplay)",
        r"(?:角色|character)(?:设定|setting)",
        r"根据(?:设定|prompt|提示)",
        r"(?:我需要|应该)(?:表现|展示|体现).*?(?:特征|性格)",
        r"这个(?:角色|character)",
    ]
    for pattern in meta_patterns:
        if re.search(pattern, think_content, re.IGNORECASE):
            issues["meta_leak"] = True
            break

    third_person_self = len(re.findall(r"玲奈(?:她|要|会|应该|需要)", think_content))
    first_person = len(re.findall(r"我(?:要|会|应该|想|觉得)", think_content))
    if third_person_self > 2 and third_person_self > first_person:
        issues["persona_break"] = True

    return issues


def detect_response_quality(response: str) -> Dict:
    """Detect quality issues in response."""
    issues = {
        "no_body_language": False,
        "excessive_kaomoji": False,
        "truncated": False,
        "format_chaos": False,
    }

    body_patterns = [
        r"[（(][^）)]*(?:耳朵|尾巴|眼睛|瞳孔|身体|手|脚|头|脸)[^）)]*[）)]",
        r"[（(][^）)]*(?:摇|晃|翘|垂|抖|蹭|靠|贴|抱)[^）)]*[）)]",
    ]
    has_body = any(re.search(p, response) for p in body_patterns)
    if not has_body and len(response) > 50:
        issues["no_body_language"] = True

    kaomoji_count = len(re.findall(r"[（(][^）)]{1,15}[）)]", response))
    kaomoji_count += len(re.findall(r"[\(][◕◔ωᴗ▽△∀へ><≧≦]{1,10}[\)]", response))
    if kaomoji_count > 6:
        issues["excessive_kaomoji"] = True

    if response.rstrip().endswith(("...", "…", "——", "、", "，")) and len(response) < 30:
        issues["truncated"] = True

    open_parens = response.count("(") + response.count("（")
    close_parens = response.count(")") + response.count("）")
    if abs(open_parens - close_parens) > 2:
        issues["format_chaos"] = True

    return issues


def extract_think_and_response(completion: str) -> Tuple[str, str]:
    """Extract thinking content and response from completion."""
    think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        response = completion[think_match.end():].strip()
    else:
        think_content = ""
        response = completion.strip()
    return think_content, response
