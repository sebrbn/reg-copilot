import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CANDIDATES = [
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash",
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemma-4-26b-a4b-it", "gemma-4-31b-it",
]

for m in CANDIDATES:
    try:
        r = client.models.generate_content(model=m, contents="Reply with the word OK.")
        print(f"WORKS  {m}: {r.text.strip()[:20]}")
    except Exception as e:
        print(f"FAILED {m}: {str(e)[:150]}")