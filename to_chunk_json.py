
import re
import json
import pymupdf
from helpers.pymupdf_rag_original import to_markdown


def remove_markdown_symbols(text):
    """
    Remove markdown-specific symbols from text.
    """
    # Remove headers (#, ##, ###, etc.)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold (**text** or __text__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove italic (*text* or _text_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove strikethrough (~~text~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    
    # Remove inline code (`text`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove code blocks (```text```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    
    # Remove blockquotes (> text)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ___, ***)
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def pdf_to_json_chunks(pdf_path):
    try:
        print(f"[DEBUG] Opening PDF: {pdf_path}")
        # open the document
        doc = pymupdf.open(pdf_path)
        print(f"[DEBUG] Document opened successfully. Page count: {doc.page_count}")
        
        # produce a single markdown string
        md = to_markdown(
            doc,
            force_text=True,      # make sure we get text even over images
            page_chunks=False,    # we only want the natural blank-line chunks
            write_images=False,
            embed_images=False
        ).strip()
        
        print(f"[DEBUG] Markdown extracted. Length: {len(md)} characters")
        print(f"[DEBUG] First 200 characters: {repr(md[:200])}")

        # split on two-or-more newlines → each chunk is one element
        raw = re.split(r'\n{2,}', md)
        print(f"[DEBUG] Split into {len(raw)} raw chunks")
        
        # strip out any purely-empty elements and remove markdown symbols
        chunks = [remove_markdown_symbols(c.strip()) for c in raw if c.strip()]
        print(f"[DEBUG] Final chunks count: {len(chunks)}")
        if chunks:
            print(f"[DEBUG] First chunk preview: {repr(chunks[0][:100])}")
        
        return chunks
    except Exception as e:
        print(f"[ERROR] Exception in pdf_to_json_chunks: {e}")
        import traceback
        traceback.print_exc()
        return []


