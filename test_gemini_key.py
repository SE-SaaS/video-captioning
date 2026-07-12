# Temporary: check GEMINI_API_KEY against the NATIVE Gemini REST API.
# Lists the models your key can call, then sends an example prompt and prints the reply.
# Run: python test_gemini_key.py     (optionally set GEMINI_TEST_MODEL to force a model)
import os
import sys

import requests

# ── Model selection ───────────────────────────────────────────────
# Set the Gemini model to use here. Leave as "" to auto-pick the first
# model your key supports. An env var GEMINI_TEST_MODEL still overrides this.
MODEL = "gemini-3.5-flash"
# ──────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("GEMINI_API_KEY is not set (check your .env).")
    sys.exit(1)

HEADERS = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}
BASE = "https://generativelanguage.googleapis.com/v1beta"

# 1) List models that support text generation.
resp = requests.get(f"{BASE}/models", headers=HEADERS, timeout=30)
if resp.status_code != 200:
    print(f"Listing models failed: HTTP {resp.status_code}")
    print(resp.text)
    sys.exit(1)

available = [
    m["name"].replace("models/", "")
    for m in resp.json().get("models", [])
    if "generateContent" in m.get("supportedGenerationMethods", [])
]
print("Available models (generateContent):")
for name in available:
    print("  -", name)

if not available:
    print("No usable models for this key.")
    sys.exit(1)

# 2) Pick a model: env override, else the MODEL constant, else first available.
model = os.environ.get("GEMINI_TEST_MODEL") or MODEL or available[0]
print(f"\nUsing model: {model}")

# 3) Send an example prompt and print the output.
PROMPT = "In one short sentence, describe a cat sitting on a warm windowsill at sunset."
body = {"contents": [{"parts": [{"text": PROMPT}]}]}
print(f"Prompt: {PROMPT}\n")

gen = requests.post(f"{BASE}/models/{model}:generateContent", headers=HEADERS, json=body, timeout=60)
print(f"HTTP {gen.status_code}")
if gen.status_code == 200:
    text = gen.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    print("Model output:")
    print(text)
else:
    print("Error response:")
    print(gen.text)
