"""Translate English text into ASL gloss using a local Llama model via Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

def load_system_prompt() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "english_asl_gloss_llama_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load system prompt from '{prompt_path}': {exc}"
        ) from exc


VIDEO_TOKEN_REPLACEMENTS = {
    "PARENTS": "MOTHER FATHER",
    "PARENT": "MOTHER FATHER",
    "GRANDPARENTS": "GRANDMOTHER GRANDFATHER",
    "GRANDPARENT": "GRANDMOTHER GRANDFATHER",
    "CHILDREN": "CHILD",
    "KIDS": "CHILD",
    "CAN NOT": "CAN'T",
    "WILL NOT": "WON'T",
    "DO NOT": "DON'T",
    "IX-me": "I",
    "IX-ME": "I",
    "IX-i": "I",
    "IX-i": "I",
    "IX-I": "I",
    "IX-you": "YOU",
    "IX-YOU": "YOU",
    "IX-he": "HE",
    "IX-HE": "HE",
    "IX-she": "SHE",
    "IX-SHE": "SHE",
    "IX-he/she": "THEY",
    "IX-HE/SHE": "THEY",
    "IX-there": "THERE",
    "IX-THERE": "THERE",
}


def clean_gloss(response: str) -> list:
    response = response.strip()
    
    # Strip markdown code blocks if present
    if response.startswith("```"):
        if "\n" in response:
            first_line, rest = response.split("\n", 1)
            if first_line.strip().startswith("```"):
                response = rest
        if response.endswith("```"):
            response = response[:-3].strip()
            
    # Try to find a JSON array in the response
    match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
    if match:
        response = match.group(0)
        
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON response from Ollama: {response}") from exc
        
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON array, got: {type(data)}")
        
    try:
        from asl_llm_video_mapping import is_gloss_in_db
    except ImportError:
        def is_gloss_in_db(g):
            return False

    IGNORE_VERBS = {"AM", "IS", "ARE", "BE", "BEEN", "BEING", "TO"}
    normalized_data = []
    for item in data:
        if not isinstance(item, dict):
            continue
        gloss = str(item.get("gloss", "")).strip().upper()
        fallback = str(item.get("fallback", "")).strip().upper()
        
        # Apply normalization replacements
        # If the user input gloss can be found in the best_asl_videos.json, then use the user input one
        if is_gloss_in_db(gloss):
            fallback = gloss
        else:
            for old_token, new_token in VIDEO_TOKEN_REPLACEMENTS.items():
                if gloss == old_token:
                    gloss = new_token
                    break
            
            if is_gloss_in_db(gloss):
                fallback = gloss
            else:
                for old_token, new_token in VIDEO_TOKEN_REPLACEMENTS.items():
                    if fallback == old_token:
                        fallback = new_token
                        break
                
        # Split tokens just in case multiple words were grouped, filtering out helper verbs
        gloss_parts = [g for g in gloss.split() if g and g not in IGNORE_VERBS]
        fallback_parts = [f for f in fallback.split() if f and f not in IGNORE_VERBS]
        
        max_len = max(len(gloss_parts), len(fallback_parts))
        for i in range(max_len):
            g_part = gloss_parts[i] if i < len(gloss_parts) else (gloss_parts[-1] if gloss_parts else "")
            f_part = fallback_parts[i] if i < len(fallback_parts) else (fallback_parts[-1] if fallback_parts else "")
            if g_part or f_part:
                normalized_data.append({
                    "gloss": g_part,
                    "fallback": f_part
                })
                
    return normalized_data


def ask_llama(text: str, model: str = DEFAULT_MODEL) -> list:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": load_system_prompt()},
            {
                "role": "user",
                "content": f"Translate this English text into a structured ASL gloss sequence with fallback synonyms:\n{text}",
            },
        ],
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is installed, running, "
            f"and that you have pulled the model '{model}'."
        ) from exc

    try:
        return clean_gloss(data["message"]["content"])
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Ollama response: {data}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate English text to ASL gloss with a local Llama model."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="English text to translate. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name. Default: {DEFAULT_MODEL}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = " ".join(args.text).strip() or sys.stdin.read().strip()

    if not text:
        print("Please provide English text as an argument or through stdin.", file=sys.stderr)
        return 2

    try:
        gloss = ask_llama(text, args.model)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(gloss, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())