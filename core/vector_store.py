import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, INDEX_METADATA_FILE

class BookVectorStore:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="books_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self._load_metadata()

    def _load_metadata(self):
        if INDEX_METADATA_FILE.exists():
            try:
                with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        INDEX_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def is_book_indexed(self, file_name: str) -> bool:
        return file_name in self.metadata

    def get_indexed_books(self) -> Dict[str, Any]:
        return self.metadata

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 200):
        """Adds text chunks to ChromaDB in batches."""
        if not chunks:
            return
        
        file_name = chunks[0]["file_name"]
        book_title = chunks[0]["book_title"]
        
        # Remove any existing chunks for this file to avoid duplicates
        self.delete_book(file_name)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [item["id"] for item in batch]
            documents = [item["text"] for item in batch]
            metadatas = [{
                "book_title": item["book_title"],
                "file_name": item["file_name"],
                "page": int(item.get("page", 1)),
                "chunk_index": int(item.get("chunk_index", 0)),
                "file_path": item["file_path"]
            } for item in batch]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        self.metadata[file_name] = {
            "book_title": book_title,
            "total_chunks": len(chunks),
            "file_name": file_name
        }
        self._save_metadata()

    def search(
        self,
        query: str,
        top_k: int = 5,
        book_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches relevant text chunks, with optional filter by book."""
        where_filter = None
        if book_filter and book_filter != "Tất cả sách":
            where_filter = {"file_name": book_filter}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else []
            distances = results["distances"][0] if "distances" in results else []
            
            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "book_title": meta.get("book_title", ""),
                    "file_name": meta.get("file_name", ""),
                    "page": meta.get("page", 1),
                    "file_path": meta.get("file_path", ""),
                    "score": round(1 - dist, 4) if dist is not None else 0.0
                })
        return formatted_results

    def delete_book(self, file_name: str):
        """Removes a book and its chunks from ChromaDB."""
        try:
            self.collection.delete(where={"file_name": file_name})
        except Exception:
            pass
        if file_name in self.metadata:
            del self.metadata[file_name]
            self._save_metadata()
