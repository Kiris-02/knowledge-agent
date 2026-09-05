import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BOOKS_DIR
from core.document_loader import load_document
from core.chunker import chunk_document_sections

def run_tests():
    print("=== KIỂM THỬ TRÍCH XUẤT TÀI LIỆU ===")
    
    test_pdf = BOOKS_DIR / "Muôn kiếp nhân sinh I.pdf"
    test_docx = BOOKS_DIR / "QUYỀN LỰC CỦA ĐỊA LÝ.docx"
    test_epub = BOOKS_DIR / "fyodor-dostoevsky_the-idiot_eva-m-martin.epub"
    
    # 1. Test PDF
    if test_pdf.exists():
        print(f"\n[1] Kiểm tra PDF: {test_pdf.name}")
        pdf_pages = load_document(test_pdf)
        print(f"-> Trích xuất thành công {len(pdf_pages)} trang")
        if pdf_pages:
            sample = pdf_pages[10]["text"][:120].replace("\n", " ") if len(pdf_pages) > 10 else pdf_pages[0]["text"][:120]
            print(f"-> Mẫu trang trích xuất: {sample}...")
            pdf_chunks = chunk_document_sections(pdf_pages[:5])
            print(f"-> Tạo thành công {len(pdf_chunks)} chunks từ 5 trang đầu")
    
    # 2. Test DOCX
    if test_docx.exists():
        print(f"\n[2] Kiểm tra DOCX: {test_docx.name}")
        docx_sections = load_document(test_docx)
        print(f"-> Trích xuất thành công {len(docx_sections)} mục")
        if docx_sections:
            sample = docx_sections[0]["text"][:120].replace("\n", " ")
            print(f"-> Mẫu đoạn trích xuất: {sample}...")
            
    # 3. Test EPUB
    if test_epub.exists():
        print(f"\n[3] Kiểm tra EPUB: {test_epub.name}")
        epub_sections = load_document(test_epub)
        print(f"-> Trích xuất thành công {len(epub_sections)} chương")
        if epub_sections:
            sample = epub_sections[0]["text"][:120].replace("\n", " ")
            print(f"-> Mẫu chương trích xuất: {sample}...")

    print("\n=== HOÀN TẤT TẤT CẢ KIỂM THỬ CƠ BẢN ===")

if __name__ == "__main__":
    run_tests()
