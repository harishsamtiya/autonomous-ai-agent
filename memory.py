"""
FAISS-based persistent memory manager for the AI Agent.
Stores and retrieves conversation history using vector embeddings
for context-aware responses across sessions.
"""

import os
import json
import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class FAISSMemoryManager:
    """Manages persistent memory using FAISS vector store."""

    def __init__(self, persist_dir: str = "./memory_store"):
        self.persist_dir = persist_dir
        self.history_file = os.path.join(persist_dir, "chat_history.json")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self._init_memory()

    def _init_memory(self):
        """Initialize or load existing FAISS index."""
        os.makedirs(self.persist_dir, exist_ok=True)

        if os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("📂 Loaded existing memory store.")
        else:
            self.vector_store = None
            print("🆕 Starting with fresh memory.")

    def store_interaction(self, user_input: str, agent_response: str):
        """Store a conversation turn in FAISS memory."""
        timestamp = datetime.datetime.now().isoformat()

        doc = Document(
            page_content=f"User: {user_input}\nAgent: {agent_response}",
            metadata={
                "timestamp": timestamp,
                "user_input": user_input,
                "type": "conversation",
            },
        )

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])

        # Save to disk
        self.vector_store.save_local(self.persist_dir)

        # Also save to JSON for easy reading
        self._append_to_json(user_input, agent_response, timestamp)

    def retrieve_relevant_context(self, query: str, k: int = 3) -> list:
        """Retrieve the most relevant past interactions for a query."""
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def show_history(self):
        """Display conversation history."""
        if not os.path.exists(self.history_file):
            print("\n📝 No conversation history yet.")
            return

        with open(self.history_file, "r") as f:
            history = json.load(f)

        print(f"\n📝 Conversation History ({len(history)} interactions):")
        print("-" * 50)
        for i, entry in enumerate(history[-5:], 1):
            print(f"\n[{entry['timestamp'][:19]}]")
            print(f"  🧑 {entry['user'][:80]}")
            print(f"  🤖 {entry['agent'][:80]}")

    def _append_to_json(self, user_input, agent_response, timestamp):
        """Append interaction to JSON history file."""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                history = json.load(f)

        history.append({
            "timestamp": timestamp,
            "user": user_input,
            "agent": agent_response,
        })

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def get_memory_stats(self) -> dict:
        """Return memory statistics."""
        stats = {
            "total_interactions": 0,
            "vector_store_size": 0,
        }
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                history = json.load(f)
            stats["total_interactions"] = len(history)

        if self.vector_store:
            stats["vector_store_size"] = self.vector_store.index.ntotal

        return stats
