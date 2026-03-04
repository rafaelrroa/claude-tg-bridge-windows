<p align="center">
  <h1 align="center">Claude Telegram Bridge</h1>
  <p align="center">
    Turn any Telegram chat into a full Claude Code workspace — with streaming, sessions, and filesystem access.
  </p>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/quick%20start-3%20steps-brightgreen" alt="Quick Start"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/single%20file-~1750%20lines-orange" alt="Single file">
</p>

> Fork of [indoor47/claude-tg-bridge](https://github.com/indoor47/claude-tg-bridge) — extended with Windows support, forum topics, interactive navigation, context-aware permissions, and security hardening.

---

A single-file Telegram bot that spawns [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) subprocesses, streams responses in real time, and gives each Telegram topic its own isolated Claude session with filesystem access.

```
You (Telegram)              Claude Bridge                Claude CLI
┌──────────┐    /clone     ┌─────────────┐  subprocess  ┌──────────┐
│  General │─────────────► │  bridge.py  │────────────► │  claude  │
│  topic   │    /cd /new   │  (asyncio)  │◄────────────┐│  --print │
└──────────┘               └─────────────┘  stream-json │  --model │
      │                          │                      └──────────┘
      ▼                          ▼                      
┌──────────┐               ┌──────────┐                 
│ myrepo   │◄──────────────│ SQLite   │  sessions,      
│ topic    │  streamed     │ sessions │  prefs, dirs     
└──────────┘  responses    └──────────┘                  
```

## Why?

Claude CLI is powerful, but it lives in your terminal. This bridge lets you:

- **Chat with Claude from anywhere** — your phone, tablet, or any device with Telegram
- **Manage multiple projects** — each Telegram topic is an isolated workspace with its own session, model, and working directory
- **Keep full CLI capabilities** — Claude can read, write, and execute code through the same tools as the CLI (Bash, Read, Edit, Grep, etc.)
- **Resume sessions across devices** — start a session on your laptop's CLI, resume it from your phone via `/resume`

No web server, no API proxy, no cloud deployment needed. Just a single Python file running alongside Claude CLI.

## Quick Start

**Prerequisites:** Python 3.11+, [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated, a [Telegram Bot token](https://t.me/BotFather).

```bash
git clone https://github.com/rafaelrroa/claude-tg-bridge-windows
cd claude-tg-bridge-windows
pip install -r requirements.txt
```

```bash
cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN and ALLOWED_USERS (your Telegram user ID)
```

```bash
python bridge.py
```

That's it. Message your bot on Telegram and start chatting with Claude.

## Features

- **Streaming responses** — Claude's output is streamed live to Telegram, flushed every 3 seconds
- **Forum topic support** — hub-and-spoke model: General topic = management hub, sub-topics = isolated Claude workspaces
- **Session management** — auto-resume, `/history` to browse past sessions, `/new` to start fresh
- **Interactive filesystem navigation** — `/cd` shows inline buttons to browse and navigate directories
- **Image support** — send photos or albums; they're batched and analyzed by Claude
- **Context-aware commands** — management commands in the hub, session commands inside topics
- **Repository cloning** — `/clone <url>` clones a repo and opens a new topic in it
- **External session attach** — `/resume <id>` picks up a Claude CLI session started elsewhere
- **Model switching** — `/haiku`, `/sonnet`, `/opus` per topic, mid-conversation
- **Access control** — allowlist by user ID and chat ID; groups blocked by default
- **Cross-platform** — Windows and Linux; systemd service unit included

## Configuration

| Variable             | Required | Default                       | Description                                         |
| -------------------- | -------- | ----------------------------- | --------------------------------------------------- |
| `TELEGRAM_BOT_TOKEN` | **Yes**  | —                             | Bot token from [@BotFather](https://t.me/BotFather) |
| `ALLOWED_USERS`      | **Yes**  | —                             | Comma-separated Telegram user IDs                   |
| `ALLOWED_CHATS`      | No       | _(empty — blocks all groups)_ | Comma-separated group/supergroup IDs                |
| `WORKING_DIR`        | No       | `~`                           | Bot root directory (top of `claude-bot:/`)          |
| `BRIDGE_DB`          | No       | `sessions.db`                 | SQLite database path                                |
| `LOG_DIR`            | No       | `.`                           | Directory for rotating log files                    |

> **Security:** If `ALLOWED_CHATS` is empty, the bot blocks all group/supergroup access. Only private chats with users in `ALLOWED_USERS` will work.

## Telegram Setup

### 1. Create the bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the token into `TELEGRAM_BOT_TOKEN` in your `.env`

### 2. Find your user ID

Send any message to [@userinfobot](https://t.me/userinfobot) — it replies with your numeric user ID. Add it to `ALLOWED_USERS`.

### 3. Enable Forum mode (recommended)

Forum mode gives each project its own topic with an isolated session, model, and working directory.

1. Create a Telegram group
2. **Settings > Edit > Topics** — enable it (the group becomes a Supergroup with Topics)
3. Add your bot as an **Administrator** with **Manage Topics** permission
4. Get the group's numeric ID:
   - Forward a message from the group to [@userinfobot](https://t.me/userinfobot), or
   - Use [@RawDataBot](https://t.me/RawDataBot) — look for `chat.id`
   - The ID is negative, e.g. `-1001234567890`
5. Add it to `ALLOWED_CHATS` in `.env` and restart the bot

## Commands

Commands are context-aware — different commands are available in different chat types.

### Main chat / General topic (management hub)

| Command         | Description                                      |
| --------------- | ------------------------------------------------ |
| `/cd [path]`    | Navigate filesystem with interactive buttons     |
| `/pwd`          | Show current directory                           |
| `/new [folder]` | Open a project folder as a new topic             |
| `/clone <url>`  | Clone a GitHub repo and open a topic in it       |
| `/resume <id>`  | Attach a CLI session (navigate with `/cd` first) |
| `/status`       | Model, session, uptime, working dir              |
| `/help`         | Command list                                     |
| `/readme`       | In-chat usage guide                              |

### Inside a topic (Claude workspace)

| Command                    | Description                         |
| -------------------------- | ----------------------------------- |
| _(any message)_            | Send to Claude — streamed back live |
| `/stop`                    | Cancel + close this topic           |
| `/history`                 | Browse and resume past sessions     |
| `/resume <id>`             | Attach an external CLI session      |
| `/haiku` `/sonnet` `/opus` | Switch model                        |
| `/pwd`                     | Show current directory              |

### Private chat

All commands from both tables are available. `/stop` cancels the process but doesn't close anything.

## Usage

### Typical workflow

```
General topic:  /cd myproject       ← navigate to a project
General topic:  /new                ← creates topic "myproject"

  myproject topic:  fix the login bug
  myproject topic:  now add tests for it
  myproject topic:  /stop           ← done — closes the topic
```

Or clone a remote repo:

```
General topic:  /clone https://github.com/user/repo

  repo topic:  explain the architecture
  repo topic:  /stop
```

### Attach an external CLI session

Start a Claude session on your machine, then pick it up from Telegram:

```
General topic:  /cd myproject
General topic:  /resume abc12345-1234-5678-9abc-def012345678

  myproject topic:  (session ready — continue where you left off)
```

### Interactive navigation

`/cd` shows an inline keyboard:

```
 claude-bot:/myproject
  3 folders, 12 files

   main.py
   README.md

[ .. ]
[  src ]
[  tests ]
[  docs ]
```

Tap a folder to navigate. The `..` button disappears at your root (`claude-bot:/`). Pagination appears for 8+ subfolders.

### Message queueing

If Claude is busy, messages are queued and processed in order. `/stop` cancels the current task and flushes the queue.

## Architecture

All logic lives in a single file (`bridge.py`, ~1750 lines). No build step, no external services.

| Component             | Detail                                                                          |
| --------------------- | ------------------------------------------------------------------------------- |
| **Concurrency**       | Atomic busy flag on asyncio's event loop — no `await` between check and set     |
| **Streaming**         | Reads `stream-json` stdout, buffers text, flushes to Telegram every 3s          |
| **Sandboxing**        | Each chat has a navigation floor in SQLite; `/cd` cannot escape it              |
| **Permissions**       | Three contexts (private / foundational / topic) detected from Telegram metadata |
| **Process isolation** | `start_new_session` (Linux) / `CREATE_NEW_PROCESS_GROUP` (Windows)              |
| **Storage**           | SQLite with two tables: `sessions` and `user_prefs`                             |

## Running as a Service (Linux)

```bash
# Adjust paths in systemd/claude-bridge.service, then:
sudo cp systemd/claude-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-bridge
```

## Contributing

Contributions are welcome. The project is a single file (`bridge.py`) — no build system or test framework to learn.

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Test manually by running `python bridge.py` and interacting via Telegram
5. Commit with a descriptive message (`git commit -m "feat: add dark mode support"`)
6. Push and open a Pull Request

**Reporting issues:** Open an issue describing the problem, expected behavior, and steps to reproduce.

## License

MIT
