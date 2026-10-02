# Local files and usage

Persistent configuration, sessions, and usage live in `~/.jarv/` (on Windows, `%USERPROFILE%\.jarv\`). Files are created as needed:

```
~/.jarv/
├── config.json                      # settings and optional API key
├── sessions.json                    # terminal → session mappings
├── sessions/
│   ├── history-<hash>.json          # conversation history
│   ├── artifacts-<hash>.json        # subagent artifacts
│   ├── reads-<hash>.json            # retained command outputs
│   ├── usage-<hash>.json            # session token usage totals
│   ├── redo-<hash>.json             # undo/redo stack
│   └── branches-<hash>.json         # alternate branches and completed asides
├── usage.jsonl                      # append-only system-wide usage ledger
├── session-titles.json               # disposable session-title cache
├── models-dev.json                  # refreshed models.dev metadata
├── model-catalog/                   # provider model and OpenRouter route caches
├── last_update_check.txt            # successful background-check timestamp
├── update_available.txt             # pending update notice, when present
└── archive/                         # archived sessions
```

The storage layer also creates `.jarv.lock` and temporary `.jarv-transaction.json` recovery journals. Windows updates/uninstalls can leave `update-result.json` or `uninstall-result.json` for the next invocation. Pasted clipboard bitmaps are saved outside this directory, under the operating system's temporary `jarv-clipboard/` folder; copied image files are referenced in place.

Project context files (`JARV.md`, `AGENTS.md`, `CLAUDE.md`) live in your repositories, not in `~/.jarv/`. Project-context ingestion reads these files; it does not generate or update them.

System-wide usage records are appended to `~/.jarv/usage.jsonl` and retained without automatic expiry. Time-window reports filter this history without deleting records; `all` includes all available history. Legacy `usage.json` records are also read, but older session totals aren't backfilled into time-window reports. Cost is request-based and grouped by provider and tier: Jarv uses provider-reported cost when available, otherwise estimates from [models.dev](https://models.dev) pricing for the provider actually serving the model — including long-context surcharges — and shows unknown or contract-priced requests separately.
