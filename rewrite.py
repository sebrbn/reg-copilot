import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash"
).split(",")]
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CACHE_FILE = Path("rewrites.json")
CACHE = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

INSTRUCTION = """You help search the RBI Master Direction on Know Your Customer (KYC).
Rewrite the user's question as a search query. Output ONE line containing:
the original question, then formal terms, synonyms and abbreviations that a
regulation would likely use for the same ideas.
Do not answer the question. Do not explain. Maximum 40 words."""


def rewrite(q):
    if q in CACHE:                      # same question -> no API call, same result
        return CACHE[q]
    for model in MODELS:
        try:
            r = client.models.generate_content(
                model=model,
                contents=q,
                config=types.GenerateContentConfig(system_instruction=INSTRUCTION, temperature=0),
            )
            out = " ".join(r.text.split())
            CACHE[q] = out
            CACHE_FILE.write_text(json.dumps(CACHE, indent=2))
            return out
        except (errors.ServerError, errors.ClientError) as e:
            code = getattr(e, "code", None)
            if code not in (404, 429) and not isinstance(e, errors.ServerError):
                raise
    return q                            # Gemini unavailable: search with the original


if __name__ == "__main__":
    import sys
    print(rewrite(" ".join(sys.argv[1:])))