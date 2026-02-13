import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore


class RAGPipeline:
    def __init__(self, url, collection_name, model_name, model_provider):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings()
        self.url = url
        self.collection_name = collection_name
        self.model_name = model_name
        self.model_provider = model_provider

    def ingest(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        vectorstore = QdrantVectorStore.from_documents(
            docs,
            self.embeddings,
            url=self.url,
            collection_name=self.collection_name,
        )

        print("Ingestion completed")

    def query(self, question):
        vectorstore = QdrantVectorStore(
            client=QdrantClient(url=self.url),
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        retriever = vectorstore.as_retriever()
        llm = init_chat_model(self.model_name, model_provider=self.model_provider)

        prompt = ChatPromptTemplate.from_template("""
            Answer using the context only.
            Context: {context}
            Question: {question}
            Display List Page number for reference.
            """)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(question)

        return result
