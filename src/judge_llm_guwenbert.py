from __future__ import annotations
import json
import os
from openai import OpenAI

MODEL_NAME = "accounts/fireworks/models/deepseek-v3p2"


def get_client():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )


def build_prompt(question: str, answer: str, chunk_text: str) -> str:
    return f"""你是一个检索评估助手。请判断下面“检索段落”对于回答问题是否有帮助。

问题：{question}
标准答案：{answer}

检索段落：
{chunk_text}

请只输出一个 JSON，对段落打分：
0 = 完全无关，不能帮助回答问题
1 = 略微相关，只有一点点关系
2 = 比较相关，提供了部分有用信息
3 = 高度相关，直接包含答案或足以回答问题

输出格式必须严格如下：
{{"score": 0, "reason": "一句很短的话"}}

不要输出任何别的内容。"""


def judge_chunk(question: str, answer: str, chunk_text: str):
    client = get_client()
    prompt = build_prompt(question, answer, chunk_text)

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = resp.choices[0].message.content.strip()

    try:
        result = json.loads(content)
        score = int(result["score"])
        reason = str(result["reason"])
    except Exception:
        return 0, f"parse_error: {content[:80]}"

    if score not in [0, 1, 2, 3]:
        return 0, f"bad_score: {score}"

    return score, reason


if __name__ == "__main__":
    q = "秦可卿和秦钟的关系？"
    a = "姐弟"
    chunk = "秦邦业向养生堂抱了一个儿子和一个女儿。女儿小名可儿，官名兼美。后来又得了秦钟。"
    print(judge_chunk(q, a, chunk))
