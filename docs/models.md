# Models and providers

Jarv discovers models through provider APIs and combines that metadata with the bundled or refreshed [models.dev](https://models.dev) catalog for prices, context/output limits, input modalities, and reasoning controls. The picker opens from cached data and refreshes in the background. Recommendations are a filtered subset of the available models; you can also enter a model ID manually. Failed discovery falls back to cached provider data, then models.dev entries, then built-in presets.

Catalog prices are read only from the entry of the provider actually serving the model; capabilities can be matched across providers, but prices are not borrowed from a different seller. Provider-reported request cost takes precedence. Jarv also has a direct OpenAI Astra pricing fallback while that model is absent from models.dev; unknown prices remain explicitly unknown.

A pruned snapshot of the catalog ships inside the package for Jarv's eight cloud providers, allowing offline estimates for entries it knows. Refreshing the model list in `/settings` also revalidates the catalog, using a conditional request when a cached ETag is available; an unchanged response transfers no catalog body. The refreshed copy lands in `~/.jarv/models-dev.json` and overlays the bundled provider entries.

To regenerate the bundled snapshot:

```bash
python scripts/update_models_dev.py
```

models.dev is community-maintained and MIT licensed. Local providers — Ollama, LM Studio, vLLM — serve whatever you installed, so they are read live and have no catalog entry.

Choose a provider and model with `jarv /setup`, or override them for one run with `--provider` and `--model`. See [Getting started](getting-started.md) for installation and API-key setup, and [Configuration](configuration.md) for saved settings.

Jarv preserves provider reasoning needed to continue tool calls. Signed thinking blocks stay with the provider that generated them; switching providers keeps ordinary conversation and tool results without sending incompatible signed blocks.

DeepSeek enables thinking by default and requires its previous reasoning when tools are available. Older chats created before Jarv retained that reasoning, or chats continued from another provider, cannot resume with DeepSeek thinking enabled. Jarv stops before sending that request and asks you to start a fresh chat with `/new` or `--new`. The old chat remains available, and Jarv does not change your reasoning setting. Explicitly selecting reasoning `none` sends DeepSeek's native thinking-disable option.
