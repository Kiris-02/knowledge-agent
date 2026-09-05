from typing import List, Dict, Any

def recursive_split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Splits text into chunks preserving semantic paragraph or sentence boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    
    def _split(s: str, seps: List[str]) -> List[str]:
        if not seps or len(s) <= chunk_size:
            return [s]
        
        sep = seps[0]
        splits = s.split(sep) if sep else list(s)
        chunks = []
        current = ""
        
        for piece in splits:
            candidate = current + (sep if current else "") + piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(piece) > chunk_size:
                    chunks.extend(_split(piece, seps[1:]))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)
    
    # Apply overlap
    final_chunks = []
    for i, c in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev = raw_chunks[i-1]
            overlap_text = prev[-chunk_overlap:]
            c = overlap_text + " " + c
        final_chunks.append(c.strip())
        
    return [c for c in final_chunks if len(c) > 20]

def chunk_document_sections(sections: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """Chunks loaded document pages into vector-ready items with full metadata."""
    all_chunks = []
    for sec in sections:
        text = sec.get("text", "")
        chunks = recursive_split_text(text, chunk_size, chunk_overlap)
        for idx, chk in enumerate(chunks):
            chunk_id = f"{sec['file_name']}_p{sec.get('page', 1)}_c{idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chk,
                "book_title": sec["book_title"],
                "file_name": sec["file_name"],
                "page": sec.get("page", 1),
                "file_path": sec["file_path"],
                "chunk_index": idx
            })
    return all_chunks
