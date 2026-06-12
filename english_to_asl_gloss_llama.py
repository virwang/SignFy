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

SYSTEM_PROMPT = """You are an ASL gloss translator.

Translate English sentences into ASL gloss only.

OUTPUT RULES
- Output exactly one line containing only the ASL gloss.
- Do not include explanations, notes, labels, markdown, alternatives, or commentary.
- Use UPPERCASE gloss tokens only.
- Separate gloss tokens with single spaces only.
- Do not use brackets, parentheses, slashes, underscores, or special annotation symbols.
- Keep gloss concise and natural.

GRAMMAR RULES
- Prefer ASL word order rather than English word order.
- Use TOPIC COMMENT structure when natural.
- Move time expressions to the beginning of the sentence.
- After a time marker is established, do not add English tense markers.
- Remove English articles when unnecessary, such as A, AN, THE.
- Remove English linking and helper verbs when not typically signed, such as AM, IS, ARE, BE, BEEN, BEING, DO, DOES, DID, and TO.
- Use natural ASL negation with NOT, NONE, NEVER, CAN'T, DON'T, or WON'T as appropriate.
- Use natural ASL question forms using WHO, WHAT, WHEN, WHERE, WHY, HOW, WHICH, or QUESTION when needed.
- Translate meaning rather than preserving English grammar.

PRONOUN RULES
- Use only these pronouns: I, ME, YOU, HE, SHE, WE, THEY, IT, THERE.
- Never use IX notation or indexing notation.

SUBJECT PLACEMENT
- After establishing a topic, place the subject pronoun before the predicate when natural.
- Prefer "HOMEWORK I HAVE" over "HOMEWORK HAVE I".
- Prefer "STORE I GO" over "STORE GO I".
- Avoid placing pronouns at the end of clauses unless required for emphasis.

VOCABULARY NORMALIZATION
- PARENTS -> MOTHER FATHER
- PARENT -> MOTHER FATHER
- GRANDPARENTS -> GRANDMOTHER GRANDFATHER
- GRANDPARENT -> GRANDMOTHER GRANDFATHER
- CHILDREN -> CHILD
- KIDS -> CHILD
- CAN NOT -> CAN'T
- WILL NOT -> WON'T
- DO NOT -> DON'T

PROPER NAMES AND FINGERSPELLING
- Fingerspell names, brands, usernames, codes, acronyms, and words that do not have a common established ASL sign.
- Represent fingerspelling with hyphens between letters.
- Example: JOHN -> J-O-H-N
- Example: NASA -> N-A-S-A
- Example: CHATGPT -> C-H-A-T-G-P-T

SEMANTIC RULES
- Translate meaning, not individual English words.
- Translate idioms by meaning rather than literally.
- Choose the most common ASL concept for the intended meaning.
- When an English word has multiple meanings, use surrounding context to select the appropriate gloss.

EXAMPLES
English: I am going to the store tomorrow.
ASL: TOMORROW STORE I GO

English: Where do you live?
ASL: YOU LIVE WHERE

English: My parents are here.
ASL: MOTHER FATHER HERE

English: John is my friend.
ASL: J-O-H-N MY FRIEND

English: I don't understand.
ASL: I UNDERSTAND NOT
"""


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


def clean_gloss(response: str) -> str:
    for marker in ("\n\n", "\nNote:", "\n(note:", "\nExplanation:", "\n("):
        response = response.split(marker, 1)[0]

    first_line = response.strip().splitlines()[0].strip()
    prefixes = ("ASL GLOSS:", "GLOSS:", "TRANSLATION:")

    for prefix in prefixes:
        if first_line.upper().startswith(prefix):
            first_line = first_line[len(prefix) :].strip()

    gloss = first_line.strip('"` ')

    for old_token, new_token in VIDEO_TOKEN_REPLACEMENTS.items():
        gloss = gloss.replace(old_token, new_token)

    gloss = re.sub(r"\[[^\]]*\]", "", gloss)
    gloss = re.sub(r"\([^)]*\)", "", gloss)
    gloss = gloss.replace("_", " ").replace("/", " ")
    gloss = re.sub(r"(?<=[A-Z]{2})-|-(?=[A-Z]{2})", " ", gloss)
    gloss = re.sub(r"\b(?:AM|IS|ARE|BE|BEEN|BEING|TO)\b", "", gloss)
    gloss = re.sub(r"\s+", " ", gloss)
    return gloss.strip()



def ask_llama(text: str, model: str = DEFAULT_MODEL) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Translate this English text to ASL gloss. "
                    "Return only the gloss line and nothing else:\n"
                    f"{text}"
                ),
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

    print(gloss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
