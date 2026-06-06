"""Shared provider and model catalog used by setup, runtime, and settings."""

PROVIDERS = {
    "openai": {
        "backend": "responses",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "label": "OpenAI",
    },
    "openrouter": {
        "backend": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "label": "OpenRouter",
    },
    "groq": {
        "backend": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "label": "Groq",
    },
    "deepseek": {
        "backend": "openai_compat",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "key_url": "https://platform.deepseek.com/api_keys",
        "label": "DeepSeek",
    },
    "together": {
        "backend": "openai_compat",
        "base_url": "https://api.together.ai/v1",
        "env_key": "TOGETHER_API_KEY",
        "key_url": "https://api.together.ai/settings/api-keys",
        "label": "Together AI",
    },
    "fireworks": {
        "backend": "openai_compat",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "key_url": "https://fireworks.ai/account/api-keys",
        "label": "Fireworks AI",
    },
    "anthropic": {
        "backend": "anthropic",
        "base_url": None,
        "env_key": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/settings/keys",
        "label": "Anthropic",
    },
    "gemini": {
        "backend": "litellm",
        "base_url": None,
        "env_key": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
        "label": "Google Gemini",
        "litellm_prefix": "gemini",
    },
    "ollama": {
        "backend": "litellm",
        "base_url": None,
        "env_key": None,
        "key_url": None,
        "label": "Ollama",
        "litellm_prefix": "ollama",
    },
    "lm_studio": {
        "backend": "openai_compat",
        "base_url": "http://localhost:1234/v1",
        "env_key": None,
        "key_url": None,
        "label": "LM Studio",
    },
    "vllm": {
        "backend": "openai_compat",
        "base_url": "http://localhost:8000/v1",
        "env_key": None,
        "key_url": None,
        "label": "vLLM",
    },
}

LOCAL_PROVIDERS = {"ollama", "lm_studio", "vllm"}

KEY_PATTERNS: dict[str, str] = {
    "openai": r"^sk-.{20,}",
    "anthropic": r"^sk-ant-.{20,}",
    "openrouter": r"^sk-or-.{20,}",
}


PROVIDER_CHOICES = [
    ("openai", "OpenAI", "gpt-5.4-mini"),
    ("openrouter", "OpenRouter (200+ models)", "tencent/hy3-preview"),
    ("anthropic", "Anthropic", "claude-sonnet-4-6"),
    ("gemini", "Google Gemini", "gemini-3-flash-preview"),
    ("groq", "Groq", "openai/gpt-oss-120b"),
    ("deepseek", "DeepSeek", "deepseek-v4-flash"),
    ("together", "Together AI", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ("fireworks", "Fireworks AI", "accounts/fireworks/models/kimi-k2p6"),
    ("ollama", "Ollama", "llama3.3"),
    ("lm_studio", "LM Studio", "local-model"),
    ("vllm", "vLLM", "local-model"),
]

PROVIDER_MODELS = {
    "openai": [
        ("gpt-5.5", "Flagship — largest, smartest"),
        ("gpt-5.4-mini", "Balanced — faster, cheaper"),
        ("gpt-5.4-nano", "Budget — smallest, fastest"),
    ],
    "anthropic": [
        ("claude-opus-4-7", "Flagship — most capable"),
        ("claude-sonnet-4-6", "Balanced — fast and capable"),
        ("claude-haiku-4-5", "Budget — fastest, cheapest"),
    ],
    "openrouter": [
        # Top 15 by weekly token usage (openrouter.ai/models?order=top-weekly)
        ("tencent/hy3-preview", "Tencent — Hunyuan H3 (Hy3) Preview"),
        ("deepseek/deepseek-v4-flash", "DeepSeek — V4 Flash"),
        ("anthropic/claude-opus-4.7", "Anthropic — Claude Opus 4.7"),
        ("anthropic/claude-sonnet-4.6", "Anthropic — Claude Sonnet 4.6"),
        ("moonshotai/kimi-k2.6", "MoonshotAI — Kimi K2.6"),
        ("google/gemini-3-flash-preview", "Google — Gemini 3 Flash Preview"),
        ("deepseek/deepseek-v3.2", "DeepSeek — V3.2"),
        ("deepseek/deepseek-v4-pro", "DeepSeek — V4 Pro"),
        ("minimax/minimax-m2.7", "MiniMax — M2.7"),
        ("openrouter/owl-alpha", "OpenRouter — Owl Alpha"),
        ("stepfun/step-3.5-flash", "StepFun — Step 3.5 Flash"),
        ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA — Nemotron 3 Super"),
        ("anthropic/claude-opus-4.6", "Anthropic — Claude Opus 4.6"),
        ("google/gemini-2.5-flash-lite", "Google — Gemini 2.5 Flash Lite"),
        ("google/gemini-2.5-flash", "Google — Gemini 2.5 Flash"),
    ],
    "gemini": [
        ("gemini-3.1-pro-preview", "Flagship — Gemini 3.1 Pro, 2M context"),
        ("gemini-3-flash-preview", "Balanced — Gemini 3 Flash"),
        ("gemini-3.1-flash-lite", "Budget — fastest, cheapest"),
    ],
    "groq": [
        ("openai/gpt-oss-120b", "Flagship — GPT OSS 120B"),
        ("llama-3.3-70b-versatile", "Balanced — Llama 3.3 70B"),
        ("llama-3.1-8b-instant", "Budget — fastest inference"),
    ],
    "deepseek": [
        ("deepseek-v4-pro", "Flagship — DeepSeek V4 Pro, 1M context"),
        ("deepseek-v4-flash", "Budget — faster, cheaper"),
    ],
    "together": [
        ("deepseek-ai/DeepSeek-V4-Pro", "Flagship — DeepSeek V4 Pro"),
        ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Balanced — Llama 4 Maverick, 1M context"),
        ("Qwen/Qwen3.5-9B", "Budget — Qwen 3.5 9B"),
    ],
    "fireworks": [
        ("accounts/fireworks/models/kimi-k2p6", "Flagship — Kimi K2.6"),
        ("accounts/fireworks/models/minimax-m2p7", "Balanced — MiniMax M2.7"),
        ("accounts/fireworks/models/qwen3-8b", "Budget — Qwen3 8B"),
    ],
}

