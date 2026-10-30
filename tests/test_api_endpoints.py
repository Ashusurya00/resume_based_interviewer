"""Tests for API endpoints."""
import pytest
import json
from fastapi.testclient import TestClient


def test_health_endpoint(client):
    """Health endpoint should return 200 OK."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_create_session(client):
    """Should create a new session and return a session_id."""
    response = client.post(
        "/api/sessions",
        json={"candidate_name": "Test Candidate", "role": "AI/ML Engineer"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["candidate_name"] == "Test Candidate"


def test_get_session(client):
    """Should retrieve session by ID."""
    # Create session first
    create_resp = client.post("/api/sessions", json={"candidate_name": "Jane Doe"})
    session_id = create_resp.json()["session_id"]

    # Retrieve it
    get_resp = client.get(f"/api/sessions/{session_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["session_id"] == session_id
    assert data["candidate_name"] == "Jane Doe"


def test_get_nonexistent_session(client):
    """Should return 404 for unknown session ID."""
    response = client.get("/api/sessions/non-existent-id-xyz")
    assert response.status_code == 404


def test_upload_invalid_file_type(client):
    """Should reject non-PDF files."""
    # Create session
    create_resp = client.post("/api/sessions", json={"candidate_name": "Test"})
    session_id = create_resp.json()["session_id"]

    # Try uploading a text file
    response = client.post(
        f"/api/sessions/{session_id}/resume",
        files={"file": ("resume.txt", b"This is a text file", "text/plain")}
    )
    assert response.status_code == 400


def test_start_interview_without_resume(client):
    """Should fail to start interview without resume uploaded."""
    create_resp = client.post("/api/sessions", json={"candidate_name": "Test"})
    session_id = create_resp.json()["session_id"]

    start_resp = client.post(
        f"/api/sessions/{session_id}/start",
        json={"role": "AI/ML Engineer"}
    )
    assert start_resp.status_code == 400
    assert "resume" in start_resp.json()["detail"].lower()


def test_report_not_ready_for_new_session(client):
    """Should return error for report on a new session."""
    create_resp = client.post("/api/sessions", json={"candidate_name": "Test"})
    session_id = create_resp.json()["session_id"]

    report_resp = client.get(f"/api/sessions/{session_id}/report")
    assert report_resp.status_code == 400
