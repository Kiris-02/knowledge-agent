from typing import Generator, List, Dict, Any, Tuple
from core.vector_store import BookVectorStore
from core.llm_manager import LLMManager

QA_SYSTEM_PROMPT = """Bạn là Trợ lý AI Chuyên gia Nghiên cứu Sách và Báo Chí (AI Knowledge Agent).
Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên các đoạn trích từ sách được cung cấp trong phần [NGỮ CẢNH TÀI LIỆU].

Nguyên tắc bắt buộc:
1. Trả lời một cách sâu sắc, chuẩn xác, văn phong học thuật nhưng dễ hiểu và mạch lạc.
2. LUÔN LUÔN ghi rõ trích dẫn nguồn cho các ý quan trọng theo định dạng: [Tên Sách, Trang/Chương X].
3. Nếu thông tin không có trong ngữ cảnh hoặc chưa đủ để khẳng định, hãy nêu rõ điều đó thay vì tự suy diễn hoặc bịa đặt.
4. Trả lời bằng tiếng Việt chuẩn mực (trừ các thuật ngữ chuyên ngành tiếng Anh/Pháp/Đức có thể để trong ngoặc).
"""

class QAAgent:
    def __init__(self, vector_store: BookVectorStore, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager

    def ask(
        self,
        query: str,
        provider: str,
        model: str,
        book_filter: str = "Tất cả sách",
        top_k: int = 5,
        history: List[Dict[str, str]] = None
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        # Retrieve chunks
        citations = self.vector_store.search(query, top_k=top_k, book_filter=book_filter)
        
        # Build context string
        context_parts = []
        for i, c in enumerate(citations, 1):
            context_parts.append(
                f"--- [TRÍCH ĐOẠN {i}] ---\n"
                f"Tác phẩm: {c['book_title']} (File: {c['file_name']}, Trang: {c['page']})\n"
                f"Nội dung: {c['text']}\n"
            )
        context_str = "\n".join(context_parts) if context_parts else "Không tìm thấy đoạn trích phù hợp trong chỉ mục."
        
        messages = []
        if history:
            messages.extend(history)
            
        user_message_with_context = (
            f"[NGỮ CẢNH TÀI LIỆU]\n{context_str}\n\n"
            f"[CÂU HỎI CỦA NGƯỜI ĐỌC]\n{query}\n\n"
            f"Hãy trả lời chi tiết và trích dẫn rõ nguồn từ ngữ cảnh trên."
        )
        messages.append({"role": "user", "content": user_message_with_context})
        
        stream_gen = self.llm_manager.stream_chat(
            provider=provider,
            model=model,
            system_prompt=QA_SYSTEM_PROMPT,
            messages=messages,
            temperature=0.3
        )
        
        return stream_gen, citations
