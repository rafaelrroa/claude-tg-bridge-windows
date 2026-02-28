# Claude Telegram Bridge

A Telegram bot that bridges users to Claude AI via the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code). It spawns Claude CLI subprocesses with streaming JSON output, manages per-user sessions in SQLite, and relays responses back to Telegram in real time.

## Features

- **Streaming responses** — replies are updated live as Claude generates text
- **Session management** — multiple named sessions per user, persisted in SQLite
- **Image support** — send photos; albums are batched into a single Claude call
- **Per-user working directories** — each user has an isolated filesystem context
- **Forum topic support** — each Supergroup topic gets its own session and model
- **Model switching** — switch between Haiku, Sonnet, and Opus mid-conversation
- **Access control** — allowlist by Telegram user ID and/or chat ID
- **Windows + Linux** — runs natively on both; systemd unit included for Linux

## Requirements

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated
- A Telegram Bot token (from [@BotFather](https://t.me/BotFather))

## Installation

```bash
git clone https://github.com/your-user/claude-tg-bridge-windows
cd claude-tg-bridge-windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set your TELEGRAM_BOT_TOKEN and ALLOWED_USERS
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Bot token from @BotFather |
| `ALLOWED_USERS` | No | `126414160` | Comma-separated Telegram user IDs |
| `ALLOWED_CHATS` | No | _(empty)_ | Comma-separated chat/group IDs to allow |
| `WORKING_DIR` | No | `~` (home) | Default working directory for Claude |
| `BRIDGE_DB` | No | `sessions.db` | SQLite database path |
| `LOG_DIR` | No | `.` | Directory for rotating log files |

## Running

```bash
python bridge.py
```

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message |
| `/stop` | Cancel the running Claude process |
| `/new` | Start a new Claude session |
| `/history` | List past sessions |
| `/status` | Show uptime and current model |
| `/haiku` | Switch to Claude Haiku |
| `/sonnet` | Switch to Claude Sonnet |
| `/opus` | Switch to Claude Opus |
| `/pwd` | Show current working directory |
| `/cd <path>` | Change working directory |
| `/ls [path]` | List directory contents |

## Running as a Service (Linux)

```bash
# Adjust paths in systemd/claude-bridge.service, then:
sudo cp systemd/claude-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-bridge
```

## Architecture

```
Telegram update → handler → per-user async queue → Claude subprocess (stream-json) → chunked Telegram reply
```

All logic lives in a single file (`bridge.py`, ~900 lines). No build step, no test suite.

**Concurrency:** Uses an atomic busy flag on asyncio's single-threaded event loop instead of locks. No `await` between checking and setting `busy`, preventing race conditions when messages arrive faster than Claude responds.

**Process isolation:** Subprocesses are spawned with `start_new_session=True`. stderr goes to DEVNULL to avoid pipe buffer overflow. `process.wait()` has a 5-second timeout with kill fallback.

## License

MIT
