# Claude Telegram Bridge

A Telegram bot that bridges users to Claude AI via the [Claude CLI](https://docs.anthropic.com/en/docs/claude-code). It spawns Claude CLI subprocesses with streaming JSON output, manages per-user sessions in SQLite, and relays responses back to Telegram in real time.

## Features

- **Streaming responses** — only Claude's final output is sent; no intermediate noise
- **Session management** — multiple named sessions per conversation, persisted in SQLite
- **Image support** — send photos; albums are batched into a single Claude call
- **Interactive filesystem navigation** — `/cd` shows inline buttons to browse directories; each chat is sandboxed to its own floor, displayed as `claude-bot:/`
- **Forum topic support** — hub-and-spoke model: General topic = management hub, sub-topics = isolated Claude workspaces
- **Context-aware permissions** — commands are scoped: management commands in the hub, session commands inside topics
- **Repository cloning** — `/clone <url>` clones a GitHub repo and immediately opens a new topic/session in it
- **External session attach** — `/resume <id>` attaches a Claude CLI session started outside the bot
- **Model switching** — switch between Haiku, Sonnet, and Opus per topic
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
| `ALLOWED_USERS` | **Yes** | — | Comma-separated Telegram user IDs allowed to use the bot |
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
3. Add your bot to the group as an **Administrator** with **Manage Topics** permission (needed to create and close topics programmatically)
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

Commands are context-aware — available commands differ between the management hub and project topics.

### Main chat / General topic (management hub)

| Command | Description |
|---|---|
| `/cd [path]` | Navigate filesystem with interactive folder buttons |
| `/pwd` | Show current directory as `claude-bot:/path` |
| `/new [folder]` | Open a project folder as a new topic. Without args, uses the current `/cd` directory |
| `/clone <url>` | Clone a GitHub repo into the root and open a new topic in it |
| `/resume <id>` | Navigate to the project with `/cd` first, then attach an external CLI session (creates a new topic) |
| `/status` | Current model, session, running state, uptime, working dir |
| `/help` | List all commands |
| `/readme` | In-chat usage guide |

### Inside a topic (Claude workspace)

| Command | Description |
|---|---|
| _(any message)_ | Send to Claude — response streamed back |
| `/stop` | Cancel the running Claude process, clear the queue, and **close this topic** (read-only) |
| `/history` | Browse past sessions for this topic; tap to resume |
| `/resume <id>` | Attach an external CLI session to this topic |
| `/haiku` | Switch to Claude Haiku |
| `/sonnet` | Switch to Claude Sonnet |
| `/opus` | Switch to Claude Opus |
| `/pwd` | Show current directory as `claude-bot:/path` |

### Private chat

All commands from both tables are available. `/stop` cancels the process but does not close anything.

## Usage Guide

### Private chat

Send any message — Claude responds with its full output. No status messages appear; only the final result is shown. All commands are available.

### Forum / Supergroup Topics — hub-and-spoke model

**General topic (management hub):**
- Navigate the filesystem with `/cd` — shown as `claude-bot:/`
- Open project topics with `/new` (uses current `/cd` dir) or `/clone <url>`
- Attach external CLI sessions with `/resume <id>` (navigate to the project first)
- Claude is **not active** here — plain messages are redirected with a hint
- Model and session commands are disabled (they belong to topics)

**Project topics (Claude workspaces):**
- Each topic has its own Claude session, model, and working directory
- The working directory is locked to the project folder (`claude-bot:/` within that topic)
- Full Claude functionality: chat, images, session history, model switching
- `/stop` cancels any running task and closes the topic (makes it read-only)

### Typical workflow

```
General topic:  /cd myproject          ← navigate to project
General topic:  /new                   ← creates topic "myproject"
  myproject topic:  ask Claude anything
  myproject topic:  /stop              ← closes topic when done
```

Or with a remote repo:

```
General topic:  /clone https://github.com/user/myrepo
  myrepo topic:  ask Claude anything
  myrepo topic:  /stop
```

### Attaching an external CLI session

If you started a Claude session outside the bot, you can attach it:

```
General topic:  /cd myproject
General topic:  /resume <session-uuid>   ← verifies session exists in ~/.claude/projects/
  myproject topic: session ready, send messages
```

Find your session UUID in the Claude CLI output or in `~/.claude/projects/<encoded-path>/`.

### Interactive filesystem navigation

`/cd` shows an inline keyboard instead of plain text:

```
📁 claude-bot:/myproject
  3 folders, 12 files

  📄 main.py
  📄 README.md

[ ⬆️ .. ]
[ 📁 src ]
[ 📁 tests ]
[ 📁 docs ]
```

- Tap a folder button to navigate into it (updates the working directory)
- The `⬆️ ..` button goes up one level — it disappears at the chat's root (`claude-bot:/`)
- If a directory has more than 8 subfolders, pagination buttons appear: `◀️ Prev · 1/N · Next ▶️`

### Session management

- Each conversation auto-resumes its last session; use `/new` to start fresh
- `/history` shows up to 15 past sessions as tappable buttons — tap one to resume it
- `/status` shows the active session ID, model, and runtime info (main chat only)

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

All logic lives in a single file (`bridge.py`, ~1750 lines). No build step, no test suite.

**Concurrency:** Atomic busy flag on asyncio's single-threaded event loop — no `await` between checking and setting `busy`, preventing race conditions when messages arrive faster than Claude responds.

**Streaming:** Reads Claude's `stream-json` stdout line-by-line, buffers text, and flushes to Telegram every 3 seconds or on tool-use events. Only the final accumulated text is delivered to the user.

**Sandboxing:** Each chat has a navigation *floor* stored in SQLite (`home_dir` column). The foundational chat's floor is `WORKING_DIR` (`claude-bot:/`); topic and private chats are locked to the folder they were opened with. `/cd` and folder buttons cannot navigate above the floor.

**Context-aware permissions:** Three chat contexts are detected from Telegram metadata (`chat.type`, `chat.is_forum`, `message.message_thread_id`): private chats (all commands), foundational/General topic (management commands), and forum sub-topics (session commands). Commands outside their allowed context return a redirect hint.

**Process isolation:** Subprocesses are spawned with `start_new_session=True` (Linux) or `CREATE_NEW_PROCESS_GROUP` (Windows). stderr goes to DEVNULL to avoid pipe buffer overflow. `process.wait()` has a 5-second timeout with kill fallback.

## License

MIT
