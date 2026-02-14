from rag_pipline import RAGPipeline


def main():
    rag_pipeline = RAGPipeline(
        qdrant_url="http://localhost:6333",
        llm_model_provider="ollama",
        llm_model_name="llama3.1",
        embed_model_provider="ollama",
        embed_model_name="nomic-embed-text",
    )

    # rag_pipeline = RAGPipeline(
    #     qdrant_url="http://localhost:6333",
    #     llm_model_provider="openai",
    #     llm_model_name="gpt-5-nano",
    #     embed_model_provider="openai",
    #     embed_model_name="text-embedding-3-small",
    # )

    # rag_pipeline.ingest("./data/pdf/AtomicHabits.pdf", collection_name="rag_book")
    answer = rag_pipeline.query(
        "how many laws and name them?", collection_name="rag_book"
    )
    print(answer)


if __name__ == "__main__":
    main()
