from typing import Generator, List, Dict, Any, Tuple, Optional
from core.vector_store import BookVectorStore
from core.llm_manager import LLMManager

DEBATE_SYSTEM_PROMPT = """Bạn là Triết gia & Nhà Phê bình Học thuật Đa góc nhìn (AI Intellectual Debate Host).
Nhiệm vụ của bạn là tổ chức và phân tích các cuộc đối thoại, so sánh và tranh biện sâu sắc giữa các tư tưởng, trường phái hoặc tác phẩm khác nhau có trong kho sách (ví dụ: Chủ nghĩa Tư bản vs Xã hội chủ nghĩa, Khắc kỷ vs Hiện sinh, Phật giáo vs Osho, Tâm lý học phương Tây vs Triết học phương Đông).

Cấu trúc câu trả lời bắt buộc:
1. 🎯 **Bối cảnh & Vấn đề Cốt lõi**: Đặt vấn đề và tại sao chủ đề này quan trọng.
2. 🏛️ **Trường phái / Tác giả A**: Quan điểm chính, lập luận và căn cứ trích dẫn.
3. 🌿 **Trường phái / Tác giả B**: Quan điểm chính, cách tiếp cận đối lập hoặc khác biệt.
4. ⚖️ **Phân tích Đối chiếu**: Điểm tương đồng bất ngờ & Điểm bất đồng then chốt.
5. 💡 **Góc nhìn Thực tiễn**: Bài học hoặc gợi mở tư duy cho độc giả hiện đại.

Luôn giữ thái độ khách quan, học thuật, trung lập và tôn trọng các hệ tư tưởng.
"""

class DebateAgent:
    def __init__(self, vector_store: BookVectorStore, llm_manager: LLMManager):
        self.vector_store = vector_store
        self.llm_manager = llm_manager

    def debate(
        self,
        topic: str,
        provider: str,
        model: str,
        book_a: Optional[str] = None,
        book_b: Optional[str] = None,
        top_k: int = 4
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        all_citations = []
        context_parts = []

        # Retrieve for Book A / Perspective A if specified
        if book_a and book_a != "Tự động phát hiện":
            c_a = self.vector_store.search(topic, top_k=top_k, book_filter=book_a)
            all_citations.extend(c_a)
            context_parts.append(f"=== TÀI LIỆU GÓC NHÌN A ({book_a}) ===")
            for c in c_a:
                context_parts.append(f"[{c['book_title']}, Trang {c['page']}]: {c['text']}")

        # Retrieve for Book B / Perspective B if specified
        if book_b and book_b != "Tự động phát hiện":
            c_b = self.vector_store.search(topic, top_k=top_k, book_filter=book_b)
            all_citations.extend(c_b)
            context_parts.append(f"\n=== TÀI LIỆU GÓC NHÌN B ({book_b}) ===")
            for c in c_b:
                context_parts.append(f"[{c['book_title']}, Trang {c['page']}]: {c['text']}")

        # If general topic without specific books, search across the library
        if not book_a or book_a == "Tự động phát hiện":
            gen_c = self.vector_store.search(topic, top_k=top_k * 2)
            all_citations.extend(gen_c)
            context_parts.append("=== TRÍCH ĐOẠN TỔNG HỢP TỪ KHO SÁCH ===")
            for c in gen_c:
                context_parts.append(f"[{c['book_title']}, Trang {c['page']}]: {c['text']}")

        context_str = "\n".join(context_parts)

        user_prompt = (
            f"[NGỮ CẢNH TRÍCH XUẤT TỪ THƯ VIỆN]\n{context_str}\n\n"
            f"[CHỦ ĐỀ TRANH BIỆN & ĐỐI THOẠI]\n{topic}\n\n"
            f"Hãy thực hiện bài phân tích đối chiếu chuyên sâu theo cấu trúc chuẩn."
        )

        stream_gen = self.llm_manager.stream_chat(
            provider=provider,
            model=model,
            system_prompt=DEBATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.4
        )

        return stream_gen, all_citations
