import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    Integer,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Fallback/Default for development if not set
    DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String, ForeignKey("threads.id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialize the database by creating tables."""
    Base.metadata.create_all(bind=engine)


def ensure_thread(thread_id: str, name: str = None):
    """Creates a thread if it doesn't exist."""
    db = SessionLocal()
    try:
        db_thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if not db_thread:
            new_thread = Thread(id=thread_id, name=name)
            db.add(new_thread)
            db.commit()
            db.refresh(new_thread)
            return new_thread
        return db_thread
    finally:
        db.close()


def add_message(thread_id: str, role: str, content: str):
    """Adds a message to the database."""
    db = SessionLocal()
    try:
        new_message = Message(thread_id=thread_id, role=role, content=content)
        db.add(new_message)
        db.commit()
        db.refresh(new_message)
        return new_message
    finally:
        db.close()


def get_messages(thread_id: str):
    """Retrieves all messages for a specific thread."""
    db = SessionLocal()
    try:
        messages = (
            db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return messages
    finally:
        db.close()
