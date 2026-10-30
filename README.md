# AI Resume-Based Technical Interviewer

A React and FastAPI application that turns a candidate's PDF resume and a selected job role into a multi-question technical interview. Questions and answer evaluations use a retrieval-augmented generation (RAG) pipeline grounded in the supplied machine-learning textbook.

## Features

- PDF resume parsing and skills extraction
- Role-aware retrieval for AI/ML Engineer, Data Scientist, MLOps Engineer, Backend Engineer, and Data Engineer roles
- Knowledge-base ingestion: PDF extraction, overlapping chunks, sentence-transformer embeddings, and a FAISS vector index
- Adaptive interviews: later questions use the candidate profile, earlier topics, and previous score
- SQLite persistence for sessions, questions, answers, feedback, and final reports
- React interface for the complete interview flow

## Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI or Gemini API key is optional. Without one, the application uses its local fallback generator and evaluator.

## Setup

1. Create your local configuration from `.env.example`. Keep `.env` private and never commit API keys.
2. Install backend packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Build the knowledge-base index. The source PDF is expected at `backend/knowledge_base/source_documents/Introduction_to_Machine_Learning_with_Python.pdf`.

   ```powershell
   python scripts/ingest_knowledge_base.py
   ```

4. Start the backend from the project root:

   ```powershell
   uvicorn backend.app.main:app --reload
   ```

5. In another terminal, start the React frontend:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

Open `http://localhost:5173`. The API documentation is at `http://localhost:8000/docs`.

## Configuration

Important values in `.env` include `OPENAI_API_KEY`, `GEMINI_API_KEY`, `LLM_MODEL`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `DATABASE_URL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K_RETRIEVAL`.

## Testing

Run the backend test suite from the project root:

```powershell
python -m pytest -q
```

The suite includes API validation plus a complete offline interview flow: resume upload, role-aware retrieval, generated questions, persisted answers, and final report creation.

## Project flow

1. Create a session and upload a PDF resume.
2. Extract the candidate profile and select a target role.
3. Retrieve role- and resume-relevant knowledge chunks through FAISS.
4. Generate a grounded question, collect and evaluate the answer, then adapt the next question.
5. Persist all data in SQLite and present the final report.
