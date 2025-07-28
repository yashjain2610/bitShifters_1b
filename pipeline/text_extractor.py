import fitz  # PyMuPDF
import os
import re

class TextExtractor:
    """
    Extracts text from PDF documents. Includes methods for extracting full page
    text and for getting specific contextual snippets following a heading.
    """
    def __init__(self, pdf_dir):
        """
        Initializes the TextExtractor.
        Args:
            pdf_dir (str): The directory where PDF files are stored.
        """
        self.pdf_dir = pdf_dir

    def get_contextual_snippet(self, pdf_filename, page_number, heading_text, num_lines=4):
        """
        Finds a heading on a specific page and extracts the subsequent lines of text
        as a contextual snippet. Continues to the next page if necessary.

        Args:
            pdf_filename (str): The name of the PDF file.
            page_number (int): The 1-based page number to start searching on.
            heading_text (str): The text of the heading to find.
            num_lines (int): The desired number of lines for the snippet.

        Returns:
            str: The contextual snippet. Returns a fallback snippet if the heading isn't found.
        """
        pdf_path = os.path.join(self.pdf_dir, pdf_filename + ".pdf")
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return ""

        snippet = ""
        try:
            doc = fitz.open(pdf_path)
            start_page_index = page_number - 1

            if not (0 <= start_page_index < len(doc)):
                print(f"Error: Page number {page_number} is out of range for {pdf_filename}")
                doc.close()
                return ""

            # --- Find Heading and Extract Initial Text ---
            page = doc.load_page(start_page_index)
            full_page_text = page.get_text("text")

            # Use regex split for a robust, case-insensitive search for the heading.
            # re.escape handles special characters in the heading text safely.
            parts = re.split(f"({re.escape(heading_text)})", full_page_text, maxsplit=1, flags=re.IGNORECASE)
            
            text_after_heading = ""
            if len(parts) > 2: # A successful split results in 3 parts: [before, heading, after]
                text_after_heading = parts[2]
            else:
                # --- Fallback Logic: Heading Not Found ---
                print(f"Warning: Heading '{heading_text}' not found on page {page_number} of {pdf_filename}. Using top of page as context.")
                text_after_heading = full_page_text # Use the whole page as the source

            # --- Collect Lines ---
            lines = text_after_heading.strip().splitlines()
            collected_lines = [line.strip() for line in lines if line.strip()][:num_lines]

            # --- Continue to Next Page if Needed ---
            current_page_index = start_page_index
            while len(collected_lines) < num_lines and (current_page_index + 1) < len(doc):
                current_page_index += 1
                next_page = doc.load_page(current_page_index)
                next_page_text = next_page.get_text("text")
                next_page_lines = next_page_text.strip().splitlines()
                
                # Filter out empty lines
                filtered_next_page_lines = [line.strip() for line in next_page_lines if line.strip()]

                lines_to_add = num_lines - len(collected_lines)
                collected_lines.extend(filtered_next_page_lines[:lines_to_add])

            snippet = "\n".join(collected_lines)
            doc.close()

        except Exception as e:
            print(f"An error occurred while extracting context from {pdf_filename}: {e}")
            return "" # Return empty string on error

        return snippet

    def extract_text(self, pdf_filename, page_number):
        """
        Extracts all text from a given page in a PDF.
        (This original method remains unchanged as it's used for the final refinement stage).
        """
        pdf_path = os.path.join(self.pdf_dir, pdf_filename)
        
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return ""
            
        try:
            doc = fitz.open(pdf_path)
            # Page numbers in PyMuPDF are 0-indexed, so subtract 1
            page_index = page_number - 1
            if 0 <= page_index < len(doc):
                page = doc.load_page(page_index)
                text = page.get_text("text")
                doc.close()
                return text
            else:
                print(f"Error: Page number {page_number} is out of range for {pdf_filename}")
                doc.close()
                return ""
        except Exception as e:
            print(f"An error occurred while extracting text from {pdf_filename}: {e}")
            return ""