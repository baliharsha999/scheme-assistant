import os
import pytest
from src.llm.chat_history import (
    save_message,
    get_latest_chat_sessions,
    get_chat_messages,
    init_chat_history_db,
)

TEST_DB = "data/test_history.db"


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_chat_history_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_save_and_retrieve_messages():
    """Verify messages are stored and retrieved chronologically."""
    thread_id = "test_thread_abc"
    save_message(thread_id, "user", "What is PM-KISAN?", title="PM-KISAN Query", db_path=TEST_DB)
    save_message(thread_id, "assistant", "PM-KISAN gives ₹6,000 per year.", db_path=TEST_DB)

    messages = get_chat_messages(thread_id, db_path=TEST_DB)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is PM-KISAN?"
    assert messages[1]["role"] == "assistant"
    assert "₹6,000" in messages[1]["content"]


def test_prune_to_latest_five_chats():
    """Verify that only the latest 5 chats are stored in SQLite."""
    # Create 8 chats
    for i in range(1, 9):
        thread_id = f"thread_{i}"
        save_message(
            thread_id,
            "user",
            f"Question for scheme {i}?",
            title=f"Scheme {i}",
            max_chats=5,
            db_path=TEST_DB
        )

    sessions = get_latest_chat_sessions(limit=10, db_path=TEST_DB)
    assert len(sessions) == 5
    # The oldest chats (thread_1, thread_2, thread_3) should be pruned
    stored_ids = {s["thread_id"] for s in sessions}
    assert "thread_8" in stored_ids
    assert "thread_7" in stored_ids
    assert "thread_6" in stored_ids
    assert "thread_5" in stored_ids
    assert "thread_4" in stored_ids
    assert "thread_1" not in stored_ids
