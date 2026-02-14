from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel, Field
import shutil
from rag_pipline import RAGPipeline

app = FastAPI()

rag_pipeline = RAGPipeline(
    qdrant_url="http://localhost:6333",
    llm_model_provider="openai",
    llm_model_name="gpt-5-nano",
    embed_model_provider="openai",
    embed_model_name="text-embedding-3-small",
)


class QueryRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    knowledgeBaseCode: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


@app.post("/ingest")
async def ingest(
    userId: str = Form(...),
    knowledgeBaseCode: str = Form(...),
    file: UploadFile = File(...),
):
    # 1. Save file locally
    local_file_path = (
        f"./data/upload/pdf/{userId}-{knowledgeBaseCode}-{file.filename.split('.')[-1]}"
    )
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag_pipeline.ingest(
        local_file_path, user_id=userId, knowledge_base_code=knowledgeBaseCode
    )
    return {"message": "Ingestion completed"}


@app.post("/query")
async def query(request: QueryRequest):
    answer = rag_pipeline.query(
        request.question,
        user_id=request.userId,
        knowledge_base_code=request.knowledgeBaseCode,
    )
    return {"answer": answer}
