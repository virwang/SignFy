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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.getenv("LLM_PROMPT_FILE", os.path.join(SCRIPT_DIR, "llm_small_prompt.txt"))

def load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load system prompt from '{PROMPT_PATH}': {exc}"
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

def fallback_to_fingerspelling(word: str, database_keys: set) -> str:
    if not word:
        return word
    if word not in database_keys:
        # 如果資料庫沒有這個字，Python 自動轉成 J-O-H-N
        word = word.replace("-", "") # avoid double hyphens
        return "-".join(list(word)) 
    return word

def clean_gloss(response: str) -> list:
    from asl_llm_video_mapping import get_valid_glosses
    database_keys = get_valid_glosses()
    
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
        
    # Strip JavaScript-style comments (// and /* */) that LLMs sometimes hallucinate into JSON
    response = re.sub(r"//.*", "", response)
    response = re.sub(r"/\*.*?\*/", "", response, flags=re.DOTALL)
        
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON response from Ollama: {response}") from exc
        
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON array, got: {type(data)}")

    normalized_data = []
    for item in data:
        if not isinstance(item, dict):
            continue
            
        primary = str(item.get("gloss", item.get("original_word", ""))).strip().upper()
        # using regex to remove numbers 
        primary = re.sub(r'\d+$', '', primary)
        if not primary:
            continue
            
        primary = fallback_to_fingerspelling(primary, database_keys)
            
        synonyms = item.get("synonyms", [])
        if not isinstance(synonyms, list):
            syn = re.sub(r'\d+$', '', str(synonyms).strip().upper())
            synonyms = [fallback_to_fingerspelling(syn, database_keys)] if syn else []
        else:
            synonyms = [fallback_to_fingerspelling(re.sub(r'\d+$', '', str(s).strip().upper()), database_keys) for s in synonyms if s]
            
        normalized_data.append({
            "gloss": primary,
            "synonyms": synonyms
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