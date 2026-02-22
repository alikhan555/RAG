from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import shutil
from rag_pipline import RAGPipeline

app = FastAPI()

rag_pipeline = RAGPipeline(
    qdrant_url="http://localhost:6333",
    llm_model_provider="openai",
    llm_model_name="gpt-5-mini",
    embed_model_provider="openai",
    embed_model_name="text-embedding-3-small",
)


class QueryRequest(BaseModel):
    threadId: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


@app.post("/ingest")
async def ingest(
    threadId: str = Form(...),
    file: UploadFile = File(...),
):
    # 1. Save file locally
    local_file_path = f"./data/upload/pdf/{threadId}-{file.filename}"
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag_pipeline.ingest(local_file_path, thread_id=threadId)
    return {"message": "Ingestion completed"}


@app.post("/query")
async def query(request: QueryRequest):
    return StreamingResponse(
        rag_pipeline.query_stream(
            request.question,
            thread_id=request.threadId,
        ),
        media_type="text/plain",
    )
