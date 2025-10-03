#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm.py
------
Tie-breaker using Ollama model `granite3.3:2b`.

Given:
- user query (str)
- 5 candidate questions (list[str])
- their candidate IDs (list[int], any identifiers you want to log)

Returns strict JSON:
  {"match":"YES","best_id":1..5}  OR  {"match":"NO"}

Notes:
- Temperature = 0 for deterministic behavior
- format="json" to force JSON output
"""

import json
import re
from typing import List, Dict, Any

import ollama

MODEL_NAME = "granite3.3:2b"  # you verified this works well

SYSTEM_INSTRUCTIONS = (
    "You are a deterministic classifier. Return ONLY valid JSON and nothing else.\n\n"
    "Task: Given a user query and 5 candidate questions from a NADRA Q/A database, "
    "decide if ANY candidate expresses the SAME intent as the user.\n\n"
    "Rules:\n"
    "- SAME intent means the candidate would answer the user's query correctly (paraphrases OK).\n"
    "- If YES, pick the single best candidate number (1..5).\n"
    "- Output EXACTLY one of:\n"
    '  {"match":"YES","best_id":<1-5>}\n'
    '  {"match":"NO"}\n'
)

USER_PROMPT_TEMPLATE = """User query: "{query}"

Candidates:
1) Q#{id1}: "{c1}"
2) Q#{id2}: "{c2}"
3) Q#{id3}: "{c3}"
4) Q#{id4}: "{c4}"
5) Q#{id5}: "{c5}"
"""


def _extract_json(s: str) -> Dict[str, Any]:
    """Try to parse strict JSON; fallback to extracting the first {...} block."""
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError(f"Model did not return valid JSON. Raw: {s!r}")


def tie_breaker_llm(user_query: str, candidates: List[str], candidate_ids: List[int]) -> Dict[str, Any]:
    """
    Decide whether any of the top-5 candidates matches the same intent.

    Args:
      user_query: str
      candidates: list[str] length 5
      candidate_ids: list[int] length 5 (for logging / context; any ints)

    Returns:
      dict: {"match":"YES","best_id":1..5} or {"match":"NO"}
    """
    if len(candidates) != 5 or len(candidate_ids) != 5:
        raise ValueError("Exactly 5 candidates and 5 candidate_ids are required.")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=user_query,
        id1=candidate_ids[0], c1=candidates[0],
        id2=candidate_ids[1], c2=candidates[1],
        id3=candidate_ids[2], c3=candidates[2],
        id4=candidate_ids[3], c4=candidates[3],
        id5=candidate_ids[4], c5=candidates[4],
    )

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
        format="json",
    )

    content = resp["message"]["content"]
    data = _extract_json(content)

    # Validate schema
    if not isinstance(data, dict) or "match" not in data:
        raise ValueError(f"Invalid LLM response JSON: {data!r}")

    match = data["match"]
    if match not in ("YES", "NO"):
        raise ValueError(f"Invalid 'match' value: {match!r}")

    if match == "YES":
        if "best_id" not in data:
            raise ValueError("Missing 'best_id' for YES result.")
        best_id = data["best_id"]
        if not isinstance(best_id, int) or not (1 <= best_id <= 5):
            raise ValueError(f"Invalid 'best_id' (must be 1..5): {best_id!r}")

    return data
