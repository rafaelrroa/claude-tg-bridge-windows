# Claude Telegram Bridge

A Telegram bot that bridges users to Claude AI via the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code). It spawns Claude CLI subprocesses with streaming JSON output, manages per-user sessions in SQLite, and relays responses back to Telegram in real time.

## Features

- **Streaming responses** — replies appear as Claude generates text, with no intermediate noise
- **Session management** — multiple named sessions per conversation, persisted in SQLite
- **Image support** — send photos; albums are batched into a single Claude call
- **Interactive filesystem navigation** — `/ls` and `/cd` show inline buttons to browse directories; each chat is sandboxed to its own floor, displayed as `claude-bot:/`
- **Forum topic support** — each Supergroup topic gets its own session, model, and working directory
- **Repository cloning** — `/clone <url>` clones a GitHub repo and immediately opens a new topic/session in it
- **Model switching** — switch between Haiku, Sonnet, and Opus mid-conversation
- **Access control** — allowlist by Telegram user ID and/or chat ID
- **Windows + Linux** — runs natively on both; systemd unit included for Linux

## Requirements

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated (`claude --version` should work)
- A Telegram Bot token (from [@BotFather](https://t.me/BotFather))

## Installation

```bash
git clone https://github.com/your-user/claude-tg-bridge-windows
cd claude-tg-bridge-windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN, ALLOWED_USERS, ALLOWED_CHATS, WORKING_DIR
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Yes** | — | Bot token from @BotFather |
| `ALLOWED_USERS` | No | `126414160` | Comma-separated Telegram user IDs allowed to use the bot |
| `ALLOWED_CHATS` | No | _(empty — blocks all groups)_ | Comma-separated group/supergroup IDs to allow |
| `WORKING_DIR` | No | `~` (home dir) | Bot root directory — the top of the `claude-bot:/` filesystem |
| `BRIDGE_DB` | No | `sessions.db` | SQLite database path |
| `LOG_DIR` | No | `.` | Directory for rotating log files |

> **Security note:** If `ALLOWED_CHATS` is empty, the bot blocks all group/supergroup access. Only private chats with users in `ALLOWED_USERS` will work.

## Running

```bash
python bridge.py
```

## Telegram Setup

### 1. Create the bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the token and set it as `TELEGRAM_BOT_TOKEN` in your `.env`

### 2. Create a Supergroup with Topics (Forum mode) — recommended

Forum mode lets each repository/project live in its own topic with isolated session, model, and working directory.

1. Create a new Telegram group
2. Open group **Settings → Edit → Topics** and enable it — the group becomes a *Supergroup with Topics*
3. Add your bot to the group as an **Administrator** (needed to create topics programmatically)
4. Get the group's numeric ID:
   - Forward any message from the group to [@userinfobot](https://t.me/userinfobot), or
   - Use [@RawDataBot](https://t.me/RawDataBot) — it shows `chat.id` for group messages
   - The ID is negative, e.g. `-1001234567890`
5. Add it to `ALLOWED_CHATS` in your `.env`:
   ```
   ALLOWED_CHATS=-1001234567890
   ```
6. Restart the bot

### 3. Find your user ID

Send any message to [@userinfobot](https://t.me/userinfobot) — it replies with your numeric user ID. Add it to `ALLOWED_USERS`.

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Show welcome message and command reference |
| `/stop` | Cancel the running Claude process |
| `/new [folder]` | Start a new session; in forum mode creates a new topic with that working directory |
| `/clone <url>` | Clone a GitHub repo into the bot root and open a new session/topic in it |
| `/history` | List past sessions with inline buttons to resume any of them |
| `/status` | Show uptime, current model, session ID, and working directory |
| `/haiku` | Switch to Claude Haiku |
| `/sonnet` | Switch to Claude Sonnet |
| `/opus` | Switch to Claude Opus |
| `/pwd` | Show current working directory as `claude-bot:/path` |
| `/cd [path]` | Navigate to a directory; without argument shows interactive folder browser |
| `/ls [path]` | List directory contents with interactive folder buttons |

## Usage Guide

### Private chat

Send any message — Claude responds with the result of its full reasoning, using the configured tools. No status messages appear; only Claude's final output is shown.

### Forum / Supergroup Topics

The **General topic** (or the group itself if it has no topics) is the **foundational chat**:
- Its filesystem root is `claude-bot:/` — the full `WORKING_DIR`
- Use `/clone <url>` or `/new <folder>` here to spin up new topics

Each created topic is isolated:
- Working directory locked to the cloned/opened folder (`claude-bot:/` within that topic)
- Separate Claude session and model
- Cannot navigate above its own root with `/cd` or the folder buttons

### Interactive filesystem navigation

`/ls` and `/cd` show an inline keyboard instead of plain text:

```
📁 claude-bot:/myproject
  3 folders, 12 files

  📄 main.py
  📄 README.md
  ...

[ ⬆️ .. ]
[ 📁 src ]
[ 📁 tests ]
[ 📁 docs ]
```

- Tap a folder button to navigate into it (updates the working directory)
- The `⬆️ ..` button goes up one level — it disappears at the chat's root (`claude-bot:/`)
- If a directory has more than 8 subfolders, pagination buttons appear: `◀️ Prev · 1/N · Next ▶️`

### Cloning a repository

From the foundational chat or a private chat:

```
/clone https://github.com/user/myrepo
```

- Clones into `WORKING_DIR/myrepo`
- In forum mode: creates a new topic named `myrepo` and sends the first message there
- In private/group mode: sets the working directory to the cloned repo
- In both cases, a new Claude session starts in the repo directory

### Session management

- Each conversation auto-resumes its last session; use `/new` to start fresh
- `/history` shows up to 15 past sessions as tappable buttons — tap one to resume it
- `/status` shows the active session ID

### Queueing

If Claude is busy, new messages are queued automatically and processed in order. Send `/stop` to cancel the current task and flush the queue.

## Running as a Service (Linux)

```bash
# Adjust paths in systemd/claude-bridge.service, then:
sudo cp systemd/claude-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-bridge
```

## Architecture

```
Telegram update → handler → per-user async queue → Claude subprocess (stream-json) → Telegram reply
```

All logic lives in a single file (`bridge.py`). No build step, no test suite.

**Concurrency:** Atomic busy flag on asyncio's single-threaded event loop — no `await` between checking and setting `busy`, preventing race conditions when messages arrive faster than Claude responds.

**Streaming:** Reads Claude's `stream-json` stdout line-by-line, buffers text, and flushes to Telegram every 3 seconds or on tool-use events. Only the final accumulated text is sent to the user.

**Sandboxing:** Each chat has a navigation *floor* stored in SQLite. The foundational chat's floor is `WORKING_DIR`; topic/private chats are locked to the folder they were opened with. `/cd` and folder buttons cannot navigate above the floor.

**Process isolation:** Subprocesses are spawned with `start_new_session=True` (Linux) or `CREATE_NEW_PROCESS_GROUP` (Windows). stderr goes to DEVNULL to avoid pipe buffer overflow. `process.wait()` has a 5-second timeout with kill fallback.

## License

MIT
