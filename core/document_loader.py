import os
from pathlib import Path
from typing import List, Dict, Any
import pymupdf
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_pdf_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF page by page."""
    if file_path.stat().st_size == 0:
        return []
    pages = []
    doc = pymupdf.open(file_path)
    book_title = file_path.stem
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "book_title": book_title,
                "file_name": file_path.name,
                "page": page_num + 1,
                "file_path": str(file_path)
            })
    doc.close()
    return pages

def extract_docx_sections(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from DOCX paragraphs."""
    if file_path.stat().st_size == 0:
        return []
    doc = docx.Document(file_path)
    book_title = file_path.stem
    full_text = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            full_text.append(t)
    
    content = "\n\n".join(full_text)
    if not content:
        return []
    
    return [{
        "text": content,
        "book_title": book_title,
        "file_name": file_path.name,
        "page": 1,
        "file_path": str(file_path)
    }]

def extract_epub_chapters(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from EPUB book."""
    if file_path.stat().st_size == 0:
        return []
    book = epub.read_epub(str(file_path))
    book_title = file_path.stem
    sections = []
    chapter_idx = 1
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator=' ').strip()
            if text and len(text) > 50:
                sections.append({
                    "text": text,
                    "book_title": book_title,
                    "file_name": file_path.name,
                    "page": chapter_idx,
                    "file_path": str(file_path)
                })
                chapter_idx += 1
    return sections

def load_document(file_path: str | Path) -> List[Dict[str, Any]]:
    """Generic document loader supporting PDF, DOCX, and EPUB."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return extract_pdf_pages(path)
        elif ext == ".docx":
            return extract_docx_sections(path)
        elif ext == ".epub":
            return extract_epub_chapters(path)
        else:
            return []
    except Exception as e:
        print(f"Bỏ qua file lỗi hoặc trống '{path.name}': {e}")
        return []
