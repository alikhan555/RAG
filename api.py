from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import shutil
from rag_pipline import RAGPipeline

app = FastAPI()

rag_pipeline = RAGPipeline(
    qdrant_url="http://localhost:6333",
    collection_name="rag_book",  # this should be in ingest method
    llm_model_provider="ollama",
    llm_model_name="llama3.1",
    embed_model_provider="ollama",
    embed_model_name="nomic-embed-text",
)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # 1. Save file locally
    local_file_path = f"./data/upload/pdf/{file.filename}"
    with open(local_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # rag_pipeline.ingest(local_file_path)
    return {"message": "Ingestion completed"}
