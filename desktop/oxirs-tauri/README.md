# OxiRS Tauri Desktop App

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Desktop GUI for OxiRS: chat interface, visual SPARQL query builder, and CAN bus monitor**

**Status**: v0.4.0 — in development on branch `0.4.0`, last verified 2026-07-19

**Tests**: 61 passed, 0 failed (`cargo nextest run -p oxirs-tauri`)

## Overview

`oxirs-tauri` is a [Tauri 2](https://tauri.app/) (`tauri` 2.11.5) desktop application that wraps a subset of OxiRS's chat, SPARQL, and industrial-connectivity functionality in a native GUI shell. The Rust backend exposes 13 `#[tauri::command]` handlers across three modules; the frontend (`ui/`) is plain HTML/CSS/JavaScript calling into them through Tauri's `window.__TAURI__.core.invoke()` bridge — there is no React/Vue/bundler.

This crate is `publish = false` and is not published to crates.io; it is a binary-only application (`src/main.rs`), not a library — see [Testing](#testing) for what that means for doc tests.

**Current implementation status**: several backend commands return mock or echo data pending integration with their real OxiRS backends — see [Implementation Status](#implementation-status). The UI shell, invoke plumbing, and the query-builder's SPARQL generation/validation/parsing logic are fully real and unit-tested; the live data sources (an `oxirs-chat` LLM provider and session store, a live `oxirs-canbus` frame stream) are not yet wired in — this is explicit in the source's own doc comments (e.g. "In this initial implementation it returns an echo for UI validation", "replace with session_store integration when available").

## Features

### Chat (`src/chat.rs` — 14 tests)
- Commands: `send_message`, `list_sessions`, `create_session`, `get_session_history`
- Sends a user message and gets a response, lists/creates chat sessions, and reads session history
- **Mock/echo mode**: `send_message` currently echoes the input back rather than routing to an LLM provider; `list_sessions`/`get_session_history` return static/empty demo data pending `oxirs-chat` session-store integration

### Visual SPARQL Query Builder (`src/query_builder.rs` — 32 tests)
- Commands: `generate_sparql`, `validate_sparql`, `parse_sparql_to_graph`, `get_example_queries`
- `generate_sparql` converts a visual graph of triple patterns, FILTER nodes, and join edges into a SPARQL `SELECT` query string
- `validate_sparql` performs a structural validity check (not full SPARQL grammar parsing)
- `parse_sparql_to_graph` round-trips a simple query's `WHERE` block back into the visual graph model via a minimal line-by-line parser — full SPARQL parsing is the `oxirs-arq` engine's job; this command is only for round-tripping simple queries in the visual builder
- `get_example_queries` returns a static list of example `(label, sparql)` pairs for the UI's "Examples" dropdown
- This module's core logic is real (not mocked) and is the most thoroughly tested part of the app

### CAN Bus Monitor (`src/canbus.rs` — 23 tests)
- Commands: `get_frames`, `get_bus_stats`, `lookup_pgn`, `get_pgn_database`, `clear_frames`
- The J1939 PGN lookup table and frame/signal view types are real
- **Mock mode**: `get_frames`/`get_bus_stats` currently return representative mock J1939 traffic instead of subscribing to a live CAN interface; the doc comments mark wiring up a real `oxirs-canbus` frame stream as a production follow-up. `clear_frames` is a no-op in mock mode

## Installation / Running

Not published to crates.io; build and run from a workspace checkout. Building the GUI bundle requires the [Tauri prerequisites](https://tauri.app/start/prerequisites/) for your OS and the `tauri-cli` (`cargo install tauri-cli --version "^2"`):

```bash
git clone https://github.com/cool-japan/oxirs
cd oxirs/desktop/oxirs-tauri
cargo tauri dev      # development, with hot reload of ui/
cargo tauri build    # production bundle
```

Or build and run the binary directly without the Tauri CLI (frontend assets are loaded from `ui/` per `tauri.conf.json`'s `frontendDist`):

```bash
cargo build --release --bin oxirs-tauri
./target/release/oxirs-tauri
```

## Project Structure

```
desktop/oxirs-tauri/
├── src/
│   ├── main.rs           # tauri::Builder + invoke_handler registration (13 commands)
│   ├── chat.rs            # Chat session commands (mock/echo backend)
│   ├── query_builder.rs   # Visual SPARQL query builder commands (real logic)
│   ├── canbus.rs           # CAN bus frame/PGN commands (mock backend)
│   └── error.rs           # AppError (Chat/NotFound/Serialization) + AppResult<T>
├── ui/                    # Vanilla HTML/CSS/JS frontend (Tauri invoke() bridge, no bundler)
│   ├── index.html
│   ├── chat.html
│   ├── query_builder.html
│   └── canbus.html
├── capabilities/          # Tauri 2 capability/permission manifest (default.json)
├── icons/                 # App bundle icons
├── gen/                   # Tauri-generated schema files
├── build.rs               # tauri-build codegen
└── tauri.conf.json        # Window (1200x800 "OxiRS Desktop"), bundle, and CSP configuration
```

## Testing

```bash
cargo nextest run -p oxirs-tauri
# 69 tests passed, 0 failed
```

All 69 tests are Rust unit tests in `#[cfg(test)] mod tests` blocks (`canbus.rs`: 23, `query_builder.rs`: 32, `chat.rs`: 14) exercising the command handlers' business logic directly. There is no `tests/` integration directory.

`oxirs-tauri` is a **binary-only crate** (`src/main.rs`, no `src/lib.rs`), so it structurally cannot have doc tests — there is no library crate root for `cargo test --doc` to run against. It is listed in the workspace-root [`.doctest-exclude`](../../.doctest-exclude) file for exactly this reason, so workspace-wide doc-test tooling skips it instead of reporting a spurious result.

## Implementation Status

| Command(s) | Status |
|---|---|
| `generate_sparql`, `validate_sparql`, `parse_sparql_to_graph`, `get_example_queries` | Real logic |
| `get_pgn_database`, `lookup_pgn` | Real, static PGN lookup table |
| `get_frames`, `get_bus_stats`, `clear_frames` | Mock data — no live `oxirs-canbus` stream wired in yet |
| `send_message` | Echoes the input — no LLM provider wired in yet |
| `list_sessions`, `create_session`, `get_session_history` | Static/empty demo data — no persistent session store yet |

## Related Tools

- [`oxirs`](../../tools/oxirs/): the CLI whose SPARQL functionality this app's query builder complements
- [`oxirs-chat`](../../ai/oxirs-chat/): the AI chat backend this app's Chat tab is designed to integrate with
- [`oxirs-canbus`](../../stream/oxirs-canbus/): the CAN bus backend this app's monitor tab is designed to integrate with

## License

Licensed under:

- Apache License, Version 2.0 ([LICENSE](../../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)

---

**OxiRS Tauri Desktop App v0.3.2** — chat UI, visual SPARQL query builder, and CAN bus monitor shell
