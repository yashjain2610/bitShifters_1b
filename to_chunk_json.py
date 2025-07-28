
import re
import json
import fitz  # PyMuPDF
from helpers.pymupdf_rag_original import to_markdown

# Define the PDF path directly in the script
PDF_PATH = r"C:\Users\Yatharth\Desktop\desktop1\AI\adobe_hack\sample_dataset\pdfs\file03.pdf"  # Change this to your actual PDF path
OUTPUT_PATH = "output_chunks_03.json"  # Change this to your desired output path

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
    # open the document
    doc = fitz.open(pdf_path)
    # produce a single markdown string
    md = to_markdown(
        doc,
        force_text=True,      # make sure we get text even over images
        page_chunks=False,    # we only want the natural blank-line chunks
        write_images=False,
        embed_images=False
    ).strip()

    # split on two-or-more newlines → each chunk is one element
    raw = re.split(r'\n{2,}', md)
    # strip out any purely-empty elements and remove markdown symbols
    chunks = [remove_markdown_symbols(c.strip()) for c in raw if c.strip()]
    return chunks

def main():
    chunks = pdf_to_json_chunks(PDF_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
