from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shutil
from rag_pipline import RAGPipeline
from database import init_db, ensure_thread, add_message, get_messages, get_threads

app = FastAPI()

# Initialize database
init_db()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, you can use ["*"] or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag_pipeline = RAGPipeline(
    qdrant_url="http://localhost:6333",
    llm_model_provider="openai",
    llm_model_name="gpt-5-nano",
    embed_model_provider="openai",
    embed_model_name="text-embedding-3-small",
)


class QueryRequest(BaseModel):
    threadId: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    isStream: bool = Field(default=False)


@app.post("/ingest")
async def ingest(
    threadId: str = Form(...),
    threadName: str = Form(...),
    file: UploadFile = File(...),
):
    # 1. Save file locally
    local_file_path = f"./data/upload/pdf/{threadId}-{file.filename}"
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Ensure thread exists in database
    ensure_thread(threadId, name=threadName)

    rag_pipeline.ingest(local_file_path, thread_id=threadId)
    return {
        "message": "Ingestion completed",
        "threadId": threadId,
        "threadName": threadName,
    }


@app.post("/query")
async def query(request: QueryRequest):
    # Store user question
    add_message(request.threadId, "user", request.question)

    if request.isStream:

        def stream_generator():
            full_response = ""
            for chunk in rag_pipeline.query_stream(
                request.question,
                thread_id=request.threadId,
            ):
                full_response += chunk
                yield chunk
            # Store assistant response after streaming completes
            add_message(request.threadId, "assistant", full_response)

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
        )
    else:
        answer = rag_pipeline.query(
            request.question,
            thread_id=request.threadId,
        )
        # Store assistant response
        add_message(request.threadId, "assistant", answer)
        return {"answer": answer}


@app.get("/messages/{threadId}")
async def get_chat_messages(threadId: str):
    messages = get_messages(threadId)
    return [
        {"role": msg.role, "content": msg.content, "timestamp": msg.created_at}
        for msg in messages
    ]


@app.get("/threads")
async def get_all_threads():
    threads = get_threads()
    return [
        {"threadId": t.id, "threadName": t.name, "createdAt": t.created_at}
        for t in threads
    ]
