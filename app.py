import os
import sys
from pathlib import Path
import streamlit as st

# Add current dir to sys.path so modules resolve cleanly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import (
    BOOKS_DIR, SUPPORTED_EXTENSIONS, DEEPSEEK_MODELS, GEMINI_MODELS, ENV_FILE
)
from core.document_loader import load_document
from core.chunker import chunk_document_sections
from core.vector_store import BookVectorStore
from core.llm_manager import LLMManager
from agents.qa_agent import QAAgent
from agents.debate_agent import DebateAgent
from agents.insight_agent import InsightAgent

st.set_page_config(
    page_title="AI Knowledge Agent - Sách & Báo Chí",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .citation-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-indexed {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-not-indexed {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = BookVectorStore()

if "llm_manager" not in st.session_state:
    st.session_state.llm_manager = LLMManager()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Scan physical files in BOOKS_DIR
def scan_library_files():
    files = []
    if BOOKS_DIR.exists():
        for p in sorted(BOOKS_DIR.glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
                is_indexed = st.session_state.vector_store.is_book_indexed(p.name)
                files.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": p,
                    "suffix": p.suffix.lower(),
                    "size_mb": size_mb,
                    "indexed": is_indexed
                })
    return files

all_files = scan_library_files()
indexed_books_meta = st.session_state.vector_store.get_indexed_books()
indexed_file_names = list(indexed_books_meta.keys())

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/books.png", width=64)
    st.title("⚙️ Cấu Hình Hệ Thống")

    # LLM Settings
    st.subheader("🤖 Nhà Cung Cấp Mô Hình")
    provider_choice = st.radio("Chọn AI Provider:", ["DeepSeek API", "Google Gemini API"], index=0)
    provider_key = "deepseek" if "DeepSeek" in provider_choice else "gemini"

    if provider_key == "deepseek":
        default_ds_key = os.getenv("DEEPSEEK_API_KEY", "")
        ds_key_input = st.text_input("DeepSeek API Key:", value=default_ds_key, type="password")
        model_choice = st.selectbox("Chọn Model:", DEEPSEEK_MODELS, index=0)
        if ds_key_input != default_ds_key and ds_key_input:
            st.session_state.llm_manager.set_keys(deepseek_api_key=ds_key_input)
    else:
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key_input = st.text_input("Google Gemini API Key:", value=default_gemini_key, type="password")
        model_choice = st.selectbox("Chọn Model:", GEMINI_MODELS, index=0)
        if gemini_key_input != default_gemini_key and gemini_key_input:
            st.session_state.llm_manager.set_keys(gemini_api_key=gemini_key_input)

    st.markdown("---")
    # File Uploader for Cloud & Local
    st.subheader("📤 Tải Lên Sách & Báo Chí")
    uploaded_files = st.file_uploader(
        "Tải file sách (PDF, DOCX, EPUB):",
        type=["pdf", "docx", "epub"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("🚀 Nạp & Lập chỉ mục tài liệu", use_container_width=True, type="primary"):
            with st.spinner("Đang xử lý và lập chỉ mục tài liệu tải lên..."):
                BOOKS_DIR.mkdir(parents=True, exist_ok=True)
                for uf in uploaded_files:
                    dest = BOOKS_DIR / uf.name
                    with open(dest, "wb") as f:
                        f.write(uf.getbuffer())
                    sec = load_document(dest)
                    chk = chunk_document_sections(sec)
                    st.session_state.vector_store.add_chunks(chk)
                st.success(f"🎉 Đã nạp thành công {len(uploaded_files)} tài liệu vào thư viện!")
                st.rerun()

    st.markdown("---")
    # Indexer Manager
    st.subheader("📚 Quản Lý Lập Chỉ Mục (RAG)")
    total_books = len(all_files)
    indexed_count = len([f for f in all_files if f["indexed"]])
    st.write(f"📊 **Tiến độ:** Đã lập chỉ mục `{indexed_count}/{total_books}` cuốn sách")

    unindexed_names = [f["name"] for f in all_files if not f["indexed"]]
    selected_to_index = st.multiselect(
        "Chọn sách cần lập chỉ mục:",
        options=[f["name"] for f in all_files],
        default=unindexed_names[:2] if unindexed_names else []
    )

    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        start_index = st.button("⚡ Lập chỉ mục", use_container_width=True)
    with col_idx2:
        refresh_btn = st.button("🔄 Quét lại", use_container_width=True)

    if refresh_btn:
        st.rerun()

    if start_index and selected_to_index:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, fname in enumerate(selected_to_index):
            file_info = next((f for f in all_files if f["name"] == fname), None)
            if file_info:
                status_text.text(f"Đang đọc: {fname}...")
                sections = load_document(file_info["path"])
                status_text.text(f"Đang cắt đoạn: {fname} ({len(sections)} trang/mục)...")
                chunks = chunk_document_sections(sections)
                status_text.text(f"Đang lưu vector: {fname} ({len(chunks)} chunks)...")
                st.session_state.vector_store.add_chunks(chunks)
            progress_bar.progress((idx + 1) / len(selected_to_index))
            
        status_text.success("🎉 Hoàn tất lập chỉ mục!")
        st.rerun()

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-header">📚 AI Knowledge Agent - Sách & Báo Chí</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trợ lý nghiên cứu, đàm thoại tư tưởng và khai phá tri thức chuyên sâu từ 70+ tác phẩm</div>', unsafe_allow_html=True)

tab_qa, tab_debate, tab_insights, tab_library = st.tabs([
    "🔍 Tra Cứu & Đối Thoại (Q&A)",
    "⚖️ Tranh Biện Đa Góc Nhìn",
    "🧠 Tóm Tắt & Sơ Đồ Tư Duy",
    "📂 Thư Viện Sách (70+ Đầu Sách)"
])

# ----------------- TAB 1: Q&A -----------------
with tab_qa:
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        st.markdown("##### 💬 Đặt câu hỏi nghiên cứu về nội dung trong sách")
    with col_q2:
        book_options = ["Tất cả sách"] + indexed_file_names
        selected_book_filter = st.selectbox("Phạm vi tìm kiếm:", book_options, index=0)

    # Display chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Xem nguồn trích dẫn đối chiếu"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"<div class='citation-box'>"
                            f"<strong>📖 {c['book_title']}</strong> (Trang: {c['page']} | Độ tương đồng: {c['score']})<br/>"
                            f"<em>\"{c['text'][:350]}...\"</em>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # Chat input
    user_query = st.chat_input("Hỏi bất kỳ điều gì (ví dụ: 'Quy luật giá trị thặng dư được giải thích ra sao?')")
    if user_query:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            qa_agent = QAAgent(st.session_state.vector_store, st.session_state.llm_manager)
            try:
                stream_gen, citations = qa_agent.ask(
                    query=user_query,
                    provider=provider_key,
                    model=model_choice,
                    book_filter=selected_book_filter,
                    top_k=5,
                    history=st.session_state.chat_history[:-1]
                )
                response_text = st.write_stream(stream_gen)
                
                if citations:
                    with st.expander("📚 Xem nguồn trích dẫn đối chiếu"):
                        for c in citations:
                            st.markdown(
                                f"<div class='citation-box'>"
                                f"<strong>📖 {c['book_title']}</strong> (Trang: {c['page']} | Độ tương đồng: {c['score']})<br/>"
                                f"<em>\"{c['text'][:350]}...\"</em>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "citations": citations
                })
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

# ----------------- TAB 2: DEBATE -----------------
with tab_debate:
    st.markdown("##### ⚖️ So Sánh Tư Tưởng & Tranh Biện Học Thuật")
    st.caption("Đối chiếu quan điểm giữa hai tác phẩm hoặc hai trường phái tư tưởng khác nhau trong kho sách.")

    c1, c2 = st.columns(2)
    with c1:
        debate_book_a = st.selectbox("Góc nhìn / Tác phẩm A:", ["Tự động phát hiện"] + indexed_file_names, index=0)
    with c2:
        debate_book_b = st.selectbox("Góc nhìn / Tác phẩm B:", ["Tự động phát hiện"] + indexed_file_names, index=0)

    suggested_topics = [
        "So sánh quan điểm về của cải và thị trường giữa Adam Smith (Wealth of Nations) và Karl Marx (Tư bản luận)",
        "So sánh cách tiếp cận nỗi khổ và tự do tâm trí giữa Triết học Khắc Kỷ và Phật giáo",
        "Tiến hóa loài người và tương lai AI theo Yuval Noah Harari (Sapiens)",
        "Tự do ý chí và sự lựa chọn cuộc đời trong 'Dám bị ghét' vs tâm lý học hiện đại"
    ]
    selected_preset = st.selectbox("Hoặc chọn chủ đề gợi ý:", ["-- Tự nhập chủ đề bên dưới --"] + suggested_topics)
    
    debate_topic_input = st.text_area(
        "Chủ đề tranh biện / câu hỏi đối chiếu:",
        value="" if selected_preset.startswith("--") else selected_preset,
        placeholder="Nhập vấn đề bạn muốn so sánh giữa các tác giả..."
    )

    if st.button("⚔️ Bắt đầu Đối thoại / Tranh biện", type="primary"):
        if not debate_topic_input.strip():
            st.warning("Vui lòng nhập chủ đề tranh biện!")
        else:
            with st.spinner("Đang tra cứu tài liệu từ cả hai phía và tổng hợp lập luận..."):
                debate_agent = DebateAgent(st.session_state.vector_store, st.session_state.llm_manager)
                try:
                    stream_gen, citations = debate_agent.debate(
                        topic=debate_topic_input,
                        provider=provider_key,
                        model=model_choice,
                        book_a=debate_book_a,
                        book_b=debate_book_b
                    )
                    st.write_stream(stream_gen)
                    if citations:
                        with st.expander("📚 Các trích đoạn tham chiếu"):
                            for c in citations:
                                st.markdown(f"- **{c['book_title']}** (Trang {c['page']}): {c['text'][:200]}...")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

# ----------------- TAB 3: INSIGHTS & MINDMAP -----------------
with tab_insights:
    st.markdown("##### 🧠 Đúc Kết Tinh Hoa & Vẽ Sơ Đồ Tư Duy (Mindmap)")
    st.caption("Rút trích các nguyên lý cốt lõi, bài học hành động và biểu diễn bằng sơ đồ Mermaid.")

    col_ins1, col_ins2 = st.columns([2, 1])
    with col_ins1:
        target_book = st.selectbox("Chọn tác phẩm cần đúc kết:", indexed_file_names if indexed_file_names else ["(Chưa có sách nào được lập chỉ mục)"])
    with col_ins2:
        aspect = st.selectbox("Khía cạnh tập trung:", ["Toàn diện tác phẩm", "Nguyên lý cốt lõi", "Bài học thực hành cuộc sống", "Triết lý nhân sinh"])

    if st.button("✨ Rút trích Tinh hoa & Tạo Sơ đồ Tư duy", type="primary"):
        if not indexed_file_names:
            st.warning("Vui lòng lập chỉ mục ít nhất một cuốn sách ở thanh bên trái trước khi tạo tóm tắt!")
        else:
            with st.spinner(f"Đang phân tích tác phẩm '{target_book}'..."):
                insight_agent = InsightAgent(st.session_state.vector_store, st.session_state.llm_manager)
                try:
                    stream_gen, citations = insight_agent.extract_insights(
                        book_file_name=target_book,
                        provider=provider_key,
                        model=model_choice,
                        focus_aspect=aspect
                    )
                    st.write_stream(stream_gen)
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

# ----------------- TAB 4: LIBRARY EXPLORER -----------------
with tab_library:
    st.markdown(f"##### 📂 Toàn bộ Kho Sách & Báo Chí ({len(all_files)} Tác Phẩm)")
    st.caption(f"Thư mục gốc: `{BOOKS_DIR}`")

    # Search filter
    search_book = st.text_input("🔍 Lọc theo tên sách:", "")
    filtered_files = [f for f in all_files if search_book.lower() in f["name"].lower()]

    for f in filtered_files:
        col_f1, col_f2, col_f3, col_f4 = st.columns([5, 2, 2, 2])
        with col_f1:
            st.markdown(f"📄 **{f['name']}**")
        with col_f2:
            st.text(f"Dung lượng: {f['size_mb']} MB")
        with col_f3:
            if f["indexed"]:
                st.markdown('<span class="badge-indexed">✓ Đã lập chỉ mục</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-not-indexed">Chưa lập chỉ mục</span>', unsafe_allow_html=True)
        with col_f4:
            if f["indexed"]:
                if st.button("Xóa chỉ mục", key=f"del_{f['name']}"):
                    st.session_state.vector_store.delete_book(f["name"])
                    st.rerun()
            else:
                if st.button("Lập chỉ mục ngay", key=f"idx_{f['name']}"):
                    with st.spinner(f"Đang lập chỉ mục {f['name']}..."):
                        sec = load_document(f["path"])
                        chk = chunk_document_sections(sec)
                        st.session_state.vector_store.add_chunks(chk)
                        st.rerun()
