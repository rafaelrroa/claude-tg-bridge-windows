# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Telegram bot that bridges users to Claude AI via the Claude CLI. Single-file Python application (`bridge.py`) that spawns Claude CLI subprocesses with streaming JSON output, manages per-user sessions in SQLite, and relays responses back to Telegram in real time.

## Running

```bash
pip install -r requirements.txt   # only dependency: python-telegram-bot>=21.0
cp .env.example .env              # configure TELEGRAM_BOT_TOKEN and ALLOWED_USERS
python bridge.py                  # starts polling
```

Environment variables: `TELEGRAM_BOT_TOKEN` (required), `ALLOWED_USERS` (comma-separated Telegram user IDs), `WORKING_DIR` (default cwd for Claude), `BRIDGE_DB` (SQLite path, default `sessions.db`), `LOG_DIR` (default `.`).

## Architecture

**Single-file design** — all logic lives in `bridge.py` (~1750 lines). No build step, no test suite.

### Key data flow

```
Telegram update → handler → per-user async queue → Claude subprocess (stream-json) → chunked Telegram reply
```

### Critical concurrency pattern

The bot uses an **atomic busy flag** on asyncio's single-threaded event loop instead of locks. The invariant is: **no `await` between checking and setting `busy`**. This prevents race conditions when multiple messages arrive before the first completes. Messages are queued and drained iteratively via `run_and_drain()`.

```python
@dataclass
class UserState:
    process: asyncio.subprocess.Process | None = None
    queue: list[str] = field(default_factory=list)
    cancelled: bool = False
    busy: bool = False
```

### Streaming response pipeline

`run_claude_streaming()` spawns `claude --print --output-format stream-json --model {model}`, reads stdout line-by-line with 0.3s timeout, parses JSON events, buffers text, and flushes to Telegram every 3 seconds or on tool-use events.

### SQLite schema

Two tables: `sessions` (user_id, session_id, model, description, timestamps, is_current) and `user_prefs` (user_id, model, working_dir, home_dir). Indexed on `(user_id, last_used DESC)`.

### Media handling

Photo albums are buffered for 1.5s to batch multiple images into a single Claude invocation. Images are saved to `/tmp/tg_images/`.

### Process lifecycle

- Subprocesses are spawned with `start_new_session=True` (process group isolation)
- stderr is sent to DEVNULL to avoid 64KB pipe buffer overflow
- `process.wait()` has a 5-second timeout with kill fallback
- Graceful shutdown kills all process groups on SIGTERM/SIGINT

## Bot commands

Commands are context-aware across three chat types:

**Main chat / General topic (management hub):** `/cd`, `/pwd`, `/new [folder]` (no args = current nav dir), `/clone <url>`, `/resume <id>`, `/status`, `/help`, `/readme`

**Forum sub-topics (Claude workspaces):** send messages to Claude, `/stop` (cancels + closes topic), `/history`, `/resume <id>`, `/haiku`, `/sonnet`, `/opus`, `/pwd`

**Private chat:** all commands available.

Commands used outside their allowed context return a redirect message. The three contexts are detected via `chat.type`, `chat.is_forum`, and `message.message_thread_id` — see `is_foundational_chat()`, `is_topic_chat()`.
