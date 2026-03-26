import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://proxy.gen-api.ru/v1/chat/completions"

# GenAPI supports 180+ models; listing popular LLMs available via OpenAI-format
MODELS = {
    "1": {
        "id": "gpt-4-1-mini",
        "name": "GPT-4.1 Mini",
        "context": "1M",
        "supports_system": True,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 32768,
        "description": "Fast & affordable GPT-4.1 tier",
    },
    "2": {
        "id": "gpt-4-1",
        "name": "GPT-4.1",
        "context": "1M",
        "supports_system": True,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 32768,
        "description": "Full GPT-4.1",
    },
    "3": {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "context": "128K",
        "supports_system": True,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 16384,
        "description": "OpenAI flagship multimodal",
    },
    "4": {
        "id": "claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "context": "200K",
        "supports_system": True,
        "temp_range": (0.0, 1.0),
        "max_tokens_limit": 8192,
        "description": "Anthropic Claude Sonnet, 200K context",
    },
    "5": {
        "id": "claude-opus-4-5",
        "name": "Claude Opus 4.5",
        "context": "200K",
        "supports_system": True,
        "temp_range": (0.0, 1.0),
        "max_tokens_limit": 8192,
        "description": "Anthropic Claude Opus, most capable",
    },
    "6": {
        "id": "gemini-2-5-flash",
        "name": "Gemini 2.5 Flash",
        "context": "1M",
        "supports_system": True,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 8192,
        "description": "Google Gemini 2.5 Flash, 1M context",
    },
    "7": {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat",
        "context": "64K",
        "supports_system": True,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 8192,
        "description": "DeepSeek V3, strong coding & reasoning",
    },
    "8": {
        "id": "deepseek-r1",
        "name": "DeepSeek R1",
        "context": "64K",
        "supports_system": False,
        "temp_range": (0.0, 2.0),
        "max_tokens_limit": 8192,
        "description": "DeepSeek R1 reasoning model, no system msg",
    },
}


def pick_model():
    print("Available models (via GenAPI OpenAI-format):")
    print(f"  {'#':<4} {'Model':<24} {'Context':<10} {'System':<8} Description")
    print("  " + "-" * 76)
    for key, m in MODELS.items():
        sys_tag = "yes" if m["supports_system"] else "no"
        print(f"  {key:<4} {m['name']:<24} {m['context']:<10} {sys_tag:<8} {m['description']}")
    choice = input("\nSelect model [1]: ").strip() or "1"
    return MODELS.get(choice, MODELS["1"])


def get_float(prompt, default, lo, hi):
    try:
        val = float(input(prompt).strip() or str(default))
        return max(lo, min(hi, val))
    except ValueError:
        return default


def get_int(prompt, default, lo, hi):
    try:
        val = int(input(prompt).strip() or str(default))
        return max(lo, min(hi, val))
    except ValueError:
        return default


def main():
    print("=== GenAPI (180+ models) ===\n")

    api_key = os.environ.get("GEN_API_KEY")
    if not api_key:
        print("Error: GEN_API_KEY not found in .env")
        sys.exit(1)

    model = pick_model()
    tlo, thi = model["temp_range"]

    print(f"\nModel: {model['name']} | Context: {model['context']} | System msg: {'yes' if model['supports_system'] else 'no'}")
    print(f"Temperature range: {tlo}–{thi} | Max output tokens: up to {model['max_tokens_limit']}\n")

    system_message = ""
    if model["supports_system"]:
        system_message = input("System message (optional, Enter to skip): ").strip()

    user_query = input("Your query: ").strip()
    if not user_query:
        print("Error: query cannot be empty")
        sys.exit(1)

    temperature = get_float(f"Temperature ({tlo}–{thi}, default 0.7): ", 0.7, tlo, thi)
    max_tokens = get_int(f"Max tokens (1–{model['max_tokens_limit']}, default 1024): ", 1024, 1, model["max_tokens_limit"])

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_query})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model["id"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print("\nSending request...\n")
    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        print("=" * 60)
        print(result["choices"][0]["message"]["content"])
        print("=" * 60)
        print(f"Model:          {result.get('model', model['id'])}")
        print(f"Finish reason:  {result['choices'][0].get('finish_reason', 'unknown')}")
        usage = result.get("usage", {})
        if usage:
            print(f"Prompt tokens:  {usage.get('prompt_tokens', '?')}")
            print(f"Output tokens:  {usage.get('completion_tokens', '?')}")
            print(f"Total tokens:   {usage.get('total_tokens', '?')}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}\n{resp.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)
    except (KeyError, IndexError) as e:
        print(f"Response parse error: {e}\n{resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
