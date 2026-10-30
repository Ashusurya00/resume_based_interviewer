"""Offline end-to-end test for the interview lifecycle."""
import fitz

from backend.app.api import dependencies
from backend.app.api.routes import interview as interview_routes
from backend.app.main import app
from backend.app.services.evaluation_service import EvaluationService
from backend.app.services.interview_service import InterviewService
from backend.app.services.question_service import QuestionService
from backend.app.services.report_service import ReportService
from backend.app.services.resume_service import ResumeService


class StaticRAGService:
    """Deterministic RAG service so the test does not need network/model downloads."""

    def build_query(self, candidate_profile, role, previous_topics=None):
        skills = ", ".join(candidate_profile.get("skills", [])[:2])
        return f"{role} retrieval for {skills}" 

    def retrieve_context(self, query):
        return [{
            "chunk_id": "test_chunk_001",
            "page": 12,
            "source_doc": "test_ml_textbook.pdf",
            "text": "Cross-validation, precision, recall, and F1 score help evaluate machine learning models."
        }]


def _resume_pdf_bytes():
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Jane Candidate\nPython, scikit-learn, TensorFlow\nMachine learning and model evaluation projects."
    )
    payload = document.tobytes()
    document.close()
    return payload


def test_complete_interview_flow(client, monkeypatch):
    interview_service = InterviewService(
        rag_service=StaticRAGService(),
        question_service=QuestionService(llm_client=None),
        evaluation_service=EvaluationService(llm_client=None),
    )

    # The start endpoint invokes its factory directly; the remaining endpoints
    # use FastAPI dependencies, so both paths are replaced with offline services.
    monkeypatch.setattr(interview_routes, "get_interview_service", lambda: interview_service)
    app.dependency_overrides[dependencies.get_interview_service] = lambda: interview_service
    app.dependency_overrides[dependencies.get_resume_service] = lambda: ResumeService(llm_client=None)
    app.dependency_overrides[dependencies.get_report_service] = lambda: ReportService(llm_client=None)

    try:
        created = client.post("/api/sessions", json={"candidate_name": "Jane Candidate"})
        assert created.status_code == 201
        session_id = created.json()["session_id"]

        uploaded = client.post(
            f"/api/sessions/{session_id}/resume",
            files={"file": ("resume.pdf", _resume_pdf_bytes(), "application/pdf")},
        )
        assert uploaded.status_code == 200
        assert "python" in [skill.lower() for skill in uploaded.json()["extracted_skills"]]

        started = client.post(f"/api/sessions/{session_id}/start", json={"role": "Data Scientist"})
        assert started.status_code == 200
        question = started.json()["current_question"]
        assert question["source_chunks"] == ["test_chunk_001"]
        assert "Data Scientist" in question["generated_query"]

        for order in range(1, 6):
            answered = client.post(
                f"/api/sessions/{session_id}/answer",
                json={"answer_text": "I use cross-validation with precision, recall, and F1 to evaluate models."},
            )
            assert answered.status_code == 200
            assert answered.json()["question_order"] == order
            if order < 5:
                next_question = client.post(f"/api/sessions/{session_id}/next")
                assert next_question.status_code == 200
                assert next_question.json()["current_question"]["source_chunks"] == ["test_chunk_001"]

        report = client.get(f"/api/sessions/{session_id}/report")
        assert report.status_code == 200
        assert report.json()["total_questions"] == 5
        assert len(report.json()["breakdown"]) == 5
    finally:
        app.dependency_overrides.clear()
