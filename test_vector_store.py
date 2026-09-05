import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.vector_store import BookVectorStore
from core.document_loader import load_document
from core.chunker import chunk_document_sections
from config import BOOKS_DIR

def test_store():
    print("=== KIỂM THỬ VECTOR STORE & CHROMADB ===")
    store = BookVectorStore()
    
    # Load 3 pages of Muôn kiếp nhân sinh
    sample_pdf = BOOKS_DIR / "Muôn kiếp nhân sinh I.pdf"
    if sample_pdf.exists():
        sections = load_document(sample_pdf)[:3]
        chunks = chunk_document_sections(sections)
        print(f"-> Tạo {len(chunks)} chunks từ '{sample_pdf.name}'...")
        store.add_chunks(chunks)
        print("-> Đã lưu vào ChromaDB thành công!")
        
        # Test query
        query = "vũ trụ và luật nhân quả"
        print(f"\n-> Truy vấn thử: '{query}'")
        results = store.search(query, top_k=2)
        for r in results:
            print(f"   [Trang {r['page']}] Score: {r['score']} - {r['text'][:100]}...")

if __name__ == "__main__":
    test_store()
