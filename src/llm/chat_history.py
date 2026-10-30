"""
Helper module for storing and retrieving conversational chat sessions
and message history in SQLite. Keeps only the latest 5 chats for student-friendly simplicity.
"""

import os
import time
import sqlite3

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints.db")
)


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


def init_chat_history_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Ensure chat_sessions and chat_messages tables exist in SQLite."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            updated_at REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            created_at REAL,
            FOREIGN KEY (thread_id) REFERENCES chat_sessions(thread_id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()


def save_message(
    thread_id: str,
    role: str,
    content: str,
    title: str | None = None,
    max_chats: int = 5,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """Save a user or assistant message and keep only the latest max_chats sessions."""
    init_chat_history_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now = time.time()

    # If title is provided or session does not exist, update title and timestamp
    if title:
        cursor.execute("""
            INSERT INTO chat_sessions (thread_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
        """, (thread_id, title[:50], now))
    else:
        cursor.execute("""
            INSERT INTO chat_sessions (thread_id, title, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                updated_at = excluded.updated_at
        """, (thread_id, "New Chat", now))

    # Insert message
    cursor.execute("""
        INSERT INTO chat_messages (thread_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (thread_id, role, content, now))

    # Prune older chat sessions so only the latest max_chats remain
    cursor.execute(f"""
        DELETE FROM chat_sessions
        WHERE thread_id NOT IN (
            SELECT thread_id FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT {int(max_chats)}
        )
    """)

    # Prune orphaned messages
    cursor.execute("""
        DELETE FROM chat_messages
        WHERE thread_id NOT IN (
            SELECT thread_id FROM chat_sessions
        )
    """)

    conn.commit()
    conn.close()


def get_latest_chat_sessions(
    limit: int = 5,
    db_path: str = DEFAULT_DB_PATH
) -> list[dict]:
    """Retrieve the latest chat sessions up to the limit (default 5)."""
    init_chat_history_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT thread_id, title, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"thread_id": row[0], "title": row[1], "updated_at": row[2]}
        for row in rows
    ]


def get_chat_messages(
    thread_id: str,
    db_path: str = DEFAULT_DB_PATH
) -> list[dict]:
    """Retrieve all messages for a specific chat thread in chronological order."""
    init_chat_history_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content
        FROM chat_messages
        WHERE thread_id = ?
        ORDER BY id ASC
    """, (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]
