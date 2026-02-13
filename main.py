from rag_pipline import RAGPipeline


def main():
    rag_pipeline = RAGPipeline(
        url="http://localhost:6333",
        collection_name="pdf_rag1",
        model_name="gpt-5-nano",
        model_provider="openai",
    )
    # rag_pipeline.ingest("./data/pdf/AtomicHabitsSmall.pdf")
    answer = rag_pipeline.query("list topic of this PDF with 1 2 3 ")
    print(answer)


if __name__ == "__main__":
    main()
