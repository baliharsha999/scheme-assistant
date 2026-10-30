import pytest
from fastapi.testclient import TestClient
from src.llm.app import app
from src.CRag.graph import graph

client = TestClient(app)


def test_home_page_get():
    """Verify home page loads HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Government Welfare Scheme Assistant" in response.text
    assert "SqliteSaver" in response.text


def test_home_post():
    """Verify legacy POST endpoint returns JSON message."""
    response = client.post("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Verify health endpoint includes SqliteSaver persistence status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["persistence"] == "SqliteSaver"


def test_empty_thread_state():
    """Verify /thread/{thread_id} returns not-persisted for new threads."""
    response = client.get("/thread/non_existent_thread_123")
    assert response.status_code == 200
    data = response.json()
    assert data["persisted"] is False


def test_persisted_thread_state():
    """Verify /thread/{thread_id} returns persisted state after saving via graph."""
    config = {"configurable": {"thread_id": "active_student_thread"}}
    graph.update_state(
        config,
        {
            "query": "What is PM-KISAN?",
            "answer": "PM-KISAN provides 6000 INR per year.",
            "abstained": False
        }
    )
    response = client.get("/thread/active_student_thread")
    assert response.status_code == 200
    data = response.json()
    assert data["persisted"] is True
    assert data["query"] == "What is PM-KISAN?"
    assert "6000" in data["answer"]
