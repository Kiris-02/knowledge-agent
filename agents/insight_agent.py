from typing import Generator, List, Dict, Any, Tuple
from core.vector_store import BookVectorStore
from core.llm_manager import LLMManager

INSIGHT_SYSTEM_PROMPT = """Bạn là Chuyên gia Khai phóng Tri thức & Đúc kết Tác phẩm (Executive Book Summarizer & Mindmapper).
Nhiệm vụ của bạn là tổng hợp các ý niệm quan trọng nhất từ tác phẩm được chỉ định, cô đọng tinh hoa và biểu diễn cấu trúc tri thức một cách trực quan.

Cấu trúc bài đúc kết bắt buộc:
1. 📖 **Tóm tắt Điều hành (Executive Summary)**: Tinh thần cốt lõi của tác phẩm trong 2-3 đoạn súc tích.
2. 💎 **5 - 7 Luận điểm & Quy luật Vàng**: Các luận điểm nền tảng, sâu sắc nhất của tác phẩm (kèm giải thích ngắn gọn).
3. 🛠️ **Hành động Thực tiễn (Actionable Takeaways)**: Cách người đọc ứng dụng các bài học này vào cuộc sống / công việc / tư duy hôm nay.
4. 🧠 **Sơ đồ Tư duy Trực quan (Mindmap)**:
BẮT BUỘC cung cấp một khối mã `mermaid` theo định dạng `mindmap` (hoặc `graph TD`), ví dụ:
```mermaid
mindmap
  root((Tên Tác Phẩm))
    Chủ Đề 1
      Ý niệm 1.1
      Ý niệm 1.2
    Chủ Đề 2
      Ý niệm 2.1
    Chủ Đề 3
      Ý niệm 3.1
```
Đảm bảo cú pháp mermaid hợp lệ và không có ký tự đặc biệt gây lỗi vẽ biểu đồ.
"""

class InsightAgent:
    def __init__(self, vector_store: BookVectorStore, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager

    def extract_insights(
        self,
        book_file_name: str,
        provider: str,
        model: str,
        focus_aspect: str = "Toàn diện"
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        # Search representative chunks
        query = f"tư tưởng cốt lõi, luận điểm chính, nguyên lý, ý nghĩa tác phẩm {focus_aspect}"
        citations = self.vector_store.search(query, top_k=8, book_filter=book_file_name)

        context_parts = [f"=== NỘI DUNG TRÍCH ĐOẠN TỪ SÁCH '{book_file_name}' ==="]
        for c in citations:
            context_parts.append(f"[Trang {c['page']}]: {c['text']}")
        context_str = "\n".join(context_parts)

        user_prompt = (
            f"[TRÍCH ĐOẠN TÁC PHẨM]\n{context_str}\n\n"
            f"[YÊU CẦU ĐÚC KẾT]\n"
            f"Tác phẩm: {book_file_name}\n"
            f"Trọng tâm: {focus_aspect}\n"
            f"Hãy biên soạn bản đúc kết tri thức và sơ đồ tư duy mermaid hoàn chỉnh."
        )

        stream_gen = self.llm_manager.stream_chat(
            provider=provider,
            model=model,
            system_prompt=INSIGHT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3
        )

        return stream_gen, citations
