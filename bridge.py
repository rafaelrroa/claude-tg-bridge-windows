#!/usr/bin/env python3
"""Claude Code Telegram Bridge v15.

Single-file Telegram bot that proxies messages to Claude CLI subprocesses.
Streaming responses, session management, image support, per-user working dirs.

v15 fixes: race conditions via atomic busy flag, iterative queue drain,
proper error handling, graceful shutdown, log rotation.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── Configuration ────────────────────────────────────────────────────

VERSION = "15.0.0"
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USERS = {
    int(x)
    for x in os.environ.get("ALLOWED_USERS", "126414160").split(",")
    if x.strip()
}
DEFAULT_WORKING_DIR = os.environ.get("WORKING_DIR", os.path.expanduser("~"))
DB_PATH = Path(os.environ.get("BRIDGE_DB", "sessions.db"))
MAX_MSG_LEN = 4000
STREAM_INTERVAL = 3  # seconds between Telegram message flushes
READLINE_TIMEOUT = 0.3  # seconds — how often to check cancellation
PROCESS_TIMEOUT = 3600  # 1 hour max per Claude invocation
IMAGE_DIR = Path("/tmp/tg_images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TOOLS = [
    "Bash", "Read", "Edit", "Write", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task(Explore)", "Task(Plan)",
]

# ── Logging ──────────────────────────────────────────────────────────

LOG_DIR = Path(os.environ.get("LOG_DIR", "."))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_DIR / "bridge.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ),
    ],
)
logger = logging.getLogger(__name__)

# ── Database ─────────────────────────────────────────────────────────


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            description TEXT,
            model TEXT DEFAULT 'sonnet',
            created_at TEXT DEFAULT (datetime('now')),
            last_used TEXT DEFAULT (datetime('now')),
            is_current INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_prefs (
            user_id TEXT PRIMARY KEY,
            model TEXT DEFAULT 'sonnet',
            working_dir TEXT
        )
    """)
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_sessions "
        "ON sessions(user_id, last_used DESC)"
    )
    conn.commit()
    conn.close()


def get_user_model(uid: str) -> str:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("SELECT model FROM user_prefs WHERE user_id = ?", (uid,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "sonnet"


def set_user_model(uid: str, model: str) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO user_prefs (user_id, model) VALUES (?, ?)",
        (uid, model),
    )
    conn.commit()
    conn.close()


def get_working_dir(uid: str) -> str:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("SELECT working_dir FROM user_prefs WHERE user_id = ?", (uid,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else DEFAULT_WORKING_DIR


def set_working_dir(uid: str, path: str) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute(
        "INSERT INTO user_prefs (user_id, working_dir) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET working_dir = ?",
        (uid, path, path),
    )
    conn.commit()
    conn.close()


def get_current_session(uid: str) -> str | None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute(
        "SELECT session_id FROM sessions WHERE user_id = ? AND is_current = 1",
        (uid,),
    )
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def set_current_session(uid: str, session_id: str, description: str = "") -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE sessions SET is_current = 0 WHERE user_id = ?", (uid,))
    c.execute(
        "SELECT id FROM sessions WHERE user_id = ? AND session_id = ?",
        (uid, session_id),
    )
    now = datetime.now(tz=timezone.utc).isoformat()
    if c.fetchone():
        c.execute(
            "UPDATE sessions SET is_current = 1, last_used = ?, "
            "description = COALESCE(NULLIF(?, ''), description) "
            "WHERE user_id = ? AND session_id = ?",
            (now, description, uid, session_id),
        )
    else:
        c.execute(
            "INSERT INTO sessions "
            "(user_id, session_id, description, is_current, model, created_at, last_used) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (uid, session_id, description, get_user_model(uid), now, now),
        )
    conn.commit()
    conn.close()


def clear_current_session(uid: str) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE sessions SET is_current = 0 WHERE user_id = ?", (uid,))
    conn.commit()
    conn.close()


def get_session_history(uid: str, limit: int = 15) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    c = conn.cursor()
    c.execute(
        "SELECT session_id, description, model, last_used, is_current "
        "FROM sessions WHERE user_id = ? ORDER BY last_used DESC LIMIT ?",
        (uid, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "desc": r[1] or "",
            "model": r[2],
            "last_used": r[3],
            "current": r[4],
        }
        for r in rows
    ]


init_db()

# ── Per-User State ───────────────────────────────────────────────────
#
# INVARIANT: The check `if state.busy` and the set `state.busy = True`
# MUST have NO `await` between them. This guarantees atomicity within
# asyncio's single-threaded event loop. Do NOT insert awaits there.


@dataclass
class UserState:
    process: asyncio.subprocess.Process | None = None
    queue: list[str] = field(default_factory=list)
    cancelled: bool = False
    busy: bool = False


user_state: dict[str, UserState] = {}
start_time = time.monotonic()

# ── Helpers ──────────────────────────────────────────────────────────

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def truncate_message(text: str, max_len: int = MAX_MSG_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        bp = text.rfind("\n", 0, max_len)
        if bp <= 0:
            bp = text.rfind(" ", 0, max_len)
        if bp <= 0:
            bp = max_len
        chunk = text[:bp].rstrip()
        if chunk:
            chunks.append(chunk)
        text = text[bp:].lstrip()
    return chunks or [""]


async def send_safe(
    update: Update, text: str, parse_mode: str | None = "Markdown"
) -> None:
    chunks = truncate_message(text)
    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=parse_mode)
        except Exception:
            clean = chunk.replace("_", "").replace("*", "").replace("`", "")
            try:
                await update.message.reply_text(clean)
            except Exception:
                await update.message.reply_text(clean[:3900] + "...(truncated)")


def is_authorized(user_id: int) -> bool:
    return user_id in ALLOWED_USERS


# ── Claude Subprocess + Streaming ────────────────────────────────────


async def run_claude_streaming(
    update: Update,
    prompt: str,
    session_id: str | None,
    model: str,
    uid: str,
    state: UserState,
) -> None:
    """Spawn claude --print and stream responses back to Telegram."""
    working_dir = get_working_dir(uid)

    cmd = [
        "claude", "--print", "--output-format", "stream-json",
        "--verbose", "--model", model,
        "--allowedTools", *ALLOWED_TOOLS,
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
        logger.info("Resuming session: %s...", session_id[:8])
    else:
        logger.info("Starting new session")
    cmd.extend(["--", prompt])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=working_dir,
        env=env,
        limit=10 * 1024 * 1024,
        start_new_session=True,
    )
    state.process = process

    new_session_id = session_id
    buffer: list[str] = []
    last_send_time = time.monotonic()

    async def flush_buffer() -> None:
        nonlocal buffer, last_send_time
        if buffer:
            text = "\n".join(buffer)
            if text.strip():
                await send_safe(update, text)
            buffer = []
            last_send_time = time.monotonic()

    try:
        while True:
            if state.cancelled:
                logger.info("Cancellation detected for %s", uid)
                await flush_buffer()
                break

            if process.returncode is not None:
                await flush_buffer()
                break

            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=READLINE_TIMEOUT
                )
            except asyncio.TimeoutError:
                if state.cancelled:
                    break
                now = time.monotonic()
                if buffer and (now - last_send_time) > STREAM_INTERVAL:
                    await flush_buffer()
                continue
            except Exception as e:
                logger.info("Read error (likely killed): %s", e)
                break

            if not line:
                break

            line_text = line.decode("utf-8", errors="replace").strip()
            if not line_text:
                continue

            try:
                data = json.loads(line_text)
                msg_type = data.get("type", "")

                if "session_id" in data and data["session_id"]:
                    new_session_id = data["session_id"]

                if msg_type == "assistant":
                    for block in data.get("message", {}).get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            buffer.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "tool")
                            await flush_buffer()
                            await update.message.reply_text(
                                f"\U0001f527 Using: {tool_name}"
                            )
            except json.JSONDecodeError:
                if line_text and not line_text.startswith("{"):
                    buffer.append(line_text)

            now = time.monotonic()
            if buffer and (now - last_send_time) > STREAM_INTERVAL:
                await flush_buffer()

    except Exception as e:
        logger.error("Streaming error: %s", e)
        await update.message.reply_text(f"Error: {str(e)[:200]}")

    finally:
        state.process = None
        with contextlib.suppress(Exception):
            await process.wait()
        await flush_buffer()

        if new_session_id:
            try:
                set_current_session(uid, new_session_id, prompt[:100])
            except Exception:
                logger.warning("Failed to save session", exc_info=True)
            await update.message.reply_text(
                f"_[Session: {new_session_id[:12]}...]_", parse_mode="Markdown"
            )


async def run_and_drain(
    update: Update,
    initial_prompt: str,
    uid: str,
    state: UserState,
) -> None:
    """Run Claude for initial_prompt, then drain queued messages iteratively."""
    prompt: str | None = initial_prompt

    while prompt is not None:
        state.cancelled = False
        model = get_user_model(uid)
        session_id = get_current_session(uid)

        status_text = f"({session_id[:8]}...)" if session_id else "(new)"
        await update.message.reply_text(f"Claude {model} {status_text}...")

        try:
            await run_claude_streaming(
                update, prompt, session_id, model, uid, state
            )
        except Exception:
            logger.exception("Claude process failed for user %s", uid)
            await update.message.reply_text("Claude process failed. Try again.")

        # Drain next from queue
        if state.cancelled or not state.queue:
            prompt = None
        else:
            prompt = state.queue.pop(0)
            await update.message.reply_text(
                f"Processing queued: {prompt[:50]}..."
            )

    # Release the busy flag — MUST be last
    state.busy = False


# ── Command Handlers ─────────────────────────────────────────────────


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Unauthorized")
        return
    uid = str(update.effective_user.id)
    model = get_user_model(uid)
    session_id = get_current_session(uid)
    history = get_session_history(uid)
    state = user_state.get(uid)
    is_running = state.busy if state else False
    wd = get_working_dir(uid)

    await update.message.reply_text(
        f"*Claude Bridge v{VERSION}*\n\n"
        f"Model: *{model}*\n"
        f"Session: `{session_id[:8] if session_id else 'None'}...`\n"
        f"Status: {'RUNNING' if is_running else 'idle'}\n"
        f"Working dir: `{wd}`\n"
        f"Sessions: {len(history)}\n\n"
        "*Commands:*\n"
        "/stop - Cancel running task\n"
        "/haiku /sonnet /opus - Switch model\n"
        "/new - New conversation\n"
        "/history - Browse sessions\n"
        "/status - Check status\n"
        "/cd <path> - Change working directory\n"
        "/pwd - Show working directory\n"
        "/ls [path] - List files",
        parse_mode="Markdown",
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return

    logger.info("/stop received from %s", uid)
    state = user_state.get(uid)

    if not state or not state.busy:
        await update.message.reply_text("Nothing running")
        return

    state.cancelled = True
    cleared = len(state.queue)
    state.queue.clear()

    process = state.process
    if process and process.returncode is None:
        try:
            pgid = os.getpgid(process.pid)
            logger.info("Killing process group %d", pgid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info("Process already dead")
        except Exception as e:
            logger.error("Error stopping process: %s", e)
            with contextlib.suppress(Exception):
                process.kill()

    msg = "Stopped."
    if cleared:
        msg += f" Cleared {cleared} queued message(s)."
    await update.message.reply_text(msg)


async def cmd_haiku(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    set_user_model(uid, "haiku")
    await update.message.reply_text("Switched to *Haiku*", parse_mode="Markdown")


async def cmd_sonnet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    set_user_model(uid, "sonnet")
    await update.message.reply_text("Switched to *Sonnet*", parse_mode="Markdown")


async def cmd_opus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    set_user_model(uid, "opus")
    await update.message.reply_text("Switched to *Opus*", parse_mode="Markdown")


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    clear_current_session(uid)
    await update.message.reply_text("Started new conversation")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return

    history = get_session_history(uid, 15)
    if not history:
        await update.message.reply_text("No session history yet")
        return

    keyboard = []
    for h in history:
        prefix = "-> " if h["current"] else ""
        desc = h["desc"][:22] or "session"
        label = f"{prefix}{desc} ({h['id'][:6]})"
        keyboard.append(
            [InlineKeyboardButton(label, callback_data=f"resume:{h['id']}")]
        )

    await update.message.reply_text(
        f"*Session History* ({len(history)} saved)\nTap to resume:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    model = get_user_model(uid)
    session_id = get_current_session(uid)
    state = user_state.get(uid)
    is_running = state.busy if state else False
    queued = len(state.queue) if state else 0
    wd = get_working_dir(uid)
    uptime = int(time.monotonic() - start_time)
    uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m"

    await update.message.reply_text(
        f"*Status*\n"
        f"Model: *{model}*\n"
        f"Session: `{session_id[:16] if session_id else 'None'}...`\n"
        f"Running: {'Yes' if is_running else 'No'}\n"
        f"Queued: {queued}\n"
        f"Working dir: `{wd}`\n"
        f"Uptime: {uptime_str}\n"
        f"Version: {VERSION}",
        parse_mode="Markdown",
    )


async def cmd_cd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    if not context.args:
        wd = get_working_dir(uid)
        await update.message.reply_text(f"Current: `{wd}`\nUsage: /cd <path>", parse_mode="Markdown")
        return
    path = " ".join(context.args)
    resolved = os.path.expanduser(path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(get_working_dir(uid), resolved)
    resolved = os.path.normpath(resolved)
    if not os.path.isdir(resolved):
        await update.message.reply_text(f"Not a directory: `{resolved}`", parse_mode="Markdown")
        return
    set_working_dir(uid, resolved)
    await update.message.reply_text(f"Working directory: `{resolved}`", parse_mode="Markdown")


async def cmd_pwd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    wd = get_working_dir(uid)
    await update.message.reply_text(f"`{wd}`", parse_mode="Markdown")


async def cmd_ls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        return
    wd = get_working_dir(uid)
    target = " ".join(context.args) if context.args else wd
    if not os.path.isabs(target):
        target = os.path.join(wd, target)
    target = os.path.normpath(target)
    if not os.path.isdir(target):
        await update.message.reply_text(f"Not a directory: `{target}`", parse_mode="Markdown")
        return
    try:
        entries = sorted(os.listdir(target))
        dirs = [f"📁 {e}/" for e in entries if os.path.isdir(os.path.join(target, e))]
        files = [f"📄 {e}" for e in entries if os.path.isfile(os.path.join(target, e))]
        result = "\n".join(dirs + files) or "(empty)"
        header = f"`{target}`\n\n"
        await send_safe(update, header + result, parse_mode="Markdown")
    except PermissionError:
        await update.message.reply_text("Permission denied")


# ── Message + Photo Handlers ─────────────────────────────────────────


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Unauthorized")
        return

    prompt = update.message.text
    if not prompt:
        return

    # Handle stop as plain text too (in case command handler is blocked)
    if prompt.lower().strip() in ("/stop", "stop", "/cancel", "cancel"):
        await cmd_stop(update, context)
        return

    state = user_state.setdefault(uid, UserState())

    # ── CRITICAL SECTION ─────────────────────────────────────────
    # No `await` between the check and set. This is atomic within
    # asyncio's single-threaded event loop.
    if state.busy:
        state.queue.append(prompt)
        pos = len(state.queue)
        await update.message.reply_text(
            f"Queued (position {pos}). Use /stop to cancel current."
        )
        return
    state.busy = True
    # ── END CRITICAL SECTION ─────────────────────────────────────

    asyncio.create_task(run_and_drain(update, prompt, uid, state))


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = str(update.effective_user.id)
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Unauthorized")
        return

    photo = update.message.photo[-1]
    caption = update.message.caption or ""

    file = await context.bot.get_file(photo.file_id)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    image_path = IMAGE_DIR / f"photo_{timestamp}_{photo.file_unique_id}.jpg"
    await file.download_to_drive(str(image_path))
    logger.info("Downloaded image to %s", image_path)

    if caption:
        prompt = (
            f"I've sent you an image saved at {image_path}. "
            f"Please read/view this image file and then: {caption}"
        )
    else:
        prompt = (
            f"I've sent you an image saved at {image_path}. "
            "Please read/view this image file and analyze what you see."
        )

    state = user_state.setdefault(uid, UserState())

    # ── CRITICAL SECTION (same pattern as handle_message) ────────
    if state.busy:
        state.queue.append(prompt)
        pos = len(state.queue)
        await update.message.reply_text(
            f"Image queued (position {pos}). Use /stop to cancel current."
        )
        return
    state.busy = True
    # ── END CRITICAL SECTION ─────────────────────────────────────

    asyncio.create_task(run_and_drain(update, prompt, uid, state))


# ── Callback Handler ─────────────────────────────────────────────────


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    uid = str(query.from_user.id)
    if not is_authorized(query.from_user.id):
        await query.answer("Unauthorized")
        return

    if query.data and query.data.startswith("resume:"):
        session_id = query.data[7:]
        set_current_session(uid, session_id)
        await query.answer("Resumed")
        await query.edit_message_text(
            f"Resumed session:\n`{session_id}`", parse_mode="Markdown"
        )


# ── Error Handler ────────────────────────────────────────────────────


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception in handler", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        with contextlib.suppress(Exception):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Internal error occurred. Please try again.",
            )


# ── Graceful Shutdown ────────────────────────────────────────────────


async def shutdown_cleanup(app: Application) -> None:
    """Kill all running Claude processes on shutdown."""
    for uid, state in user_state.items():
        if state.process and state.process.returncode is None:
            logger.info("Shutting down: killing process for user %s", uid)
            with contextlib.suppress(Exception):
                pgid = os.getpgid(state.process.pid)
                os.killpg(pgid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(state.process.wait(), timeout=3.0)
            with contextlib.suppress(Exception):
                state.process.kill()
    logger.info("All processes terminated. Goodbye.")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        raise SystemExit(1)

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .post_shutdown(shutdown_cleanup)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("haiku", cmd_haiku))
    app.add_handler(CommandHandler("sonnet", cmd_sonnet))
    app.add_handler(CommandHandler("opus", cmd_opus))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cd", cmd_cd))
    app.add_handler(CommandHandler("pwd", cmd_pwd))
    app.add_handler(CommandHandler("ls", cmd_ls))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    app.add_error_handler(error_handler)

    logger.info("Starting Claude Bridge v%s (Streaming + Images)...", VERSION)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
