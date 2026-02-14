from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from embedding_generator import EmbeddingGenerator


class RAGPipeline:
    def __init__(
        self,
        qdrant_url,
        llm_model_provider,
        llm_model_name,
        embed_model_provider,
        embed_model_name,
        **embed_kwargs,
    ):
        embed_gen = EmbeddingGenerator(
            provider=embed_model_provider,
            model=embed_model_name,
            **embed_kwargs,
        )
        self.embeddings = embed_gen.get_embeddings()
        self.qdrant_url = qdrant_url
        self.llm_model_name = llm_model_name
        self.llm_model_provider = llm_model_provider

    def ingest(self, file_path, user_id, knowledge_base_code, collection_name="RAG"):
        print(f"Loading pdf file: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print("Loaded pdf file. Total Documents: {documents}.", len(documents))

        print("Splitting pdf file into chunks.")
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=256,
            chunk_overlap=40,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        docs = text_splitter.split_documents(documents)

        # Add metadata for filtering
        for doc in docs:
            doc.metadata["user_id"] = user_id
            doc.metadata["knowledge_base_code"] = knowledge_base_code

        print(f"Splitted pdf file into chunks. Total Chunks: {len(docs)}")

        print(f"Storing chunks into vector store collection: {collection_name}")
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=collection_name,
        )
        print("Stored chunks into vector store.")

        print("Ingestion completed.")

    def query(self, question, user_id, knowledge_base_code, collection_name="RAG"):
        vectorstore = QdrantVectorStore(
            client=QdrantClient(url=self.qdrant_url),
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        # Define metadata filter
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.user_id", match=models.MatchValue(value=user_id)
                ),
                models.FieldCondition(
                    key="metadata.knowledge_base_code",
                    match=models.MatchValue(value=knowledge_base_code),
                ),
            ]
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": qdrant_filter}
        )

        llm = init_chat_model(
            self.llm_model_name, model_provider=self.llm_model_provider
        )

        prompt = ChatPromptTemplate.from_template("""
            Answer the question using only the provided context.
            If the answer is not present in the context, respond with "Answer not found in context." and do not make assumptions.
            Context: {context}
            Question: {question}
            Display list page numbers for reference if applicable.
            """)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(question)

        return result
