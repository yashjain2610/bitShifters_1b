# main.py
import json
import os
from datetime import datetime
from pipeline.text_extractor import TextExtractor
from pipeline.text_refiner import TextRefiner

def generate_summaries(ranked_sections, pdf_dir, models_dir="models"):
    """
    Generates refined text summaries for a given list of ranked document sections.
    This version is updated to summarize a targeted snippet of text following
    the heading, rather than the whole page.

    Args:
        ranked_sections (list): A list of dictionaries, where each dictionary
                                represents a ranked section. Expected keys are:
                                'document', 'page_number', and 'section_title'.
        pdf_dir (str): The path to the directory containing the PDF files.
        models_dir (str): The path to the directory where the T5 model is stored.

    Returns:
        list: A list of dictionaries, each containing the refined summary
              and original document info.
    """
    print("--- Starting Focused Summary Generation ---")
    
    # --- 1. Initialize Components ---
    try:
        extractor = TextExtractor(pdf_dir)
        refiner = TextRefiner(models_dir)
    except FileNotFoundError as e:
        print(f"Error initializing components: {e}")
        print("Please ensure you have run 'python3 download_models.py' and that the model/PDF paths are correct.")
        return []

    # --- 2. Process Each Section ---
    refined_results = []
    for i, section in enumerate(ranked_sections):
        title = section.get('section_title', 'N/A')
        print(f"\nProcessing section {i+1}/{len(ranked_sections)}: '{title}' from '{section['document']}'...")
        
        # --- MODIFICATION ---
        # Instead of getting the whole page, we now extract a targeted snippet
        # of 20 lines that follows the specific heading. This provides a much
        # more relevant context for the summarizer.
        text_for_summarization = extractor.get_contextual_snippet(
            pdf_filename=section['document'],
            page_number=section['page_number'],
            heading_text=title,
            num_lines=50  # Get 20 lines of context as requested
        )
        
        if text_for_summarization:
            # Refine the extracted snippet using the T5 model
            print(f"  -> Snippet extracted. Generating summary...")
            refined_text = refiner.refine(text_for_summarization)
            refined_results.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
            print(f"  -> Summary generated successfully.")
        else:
            print(f"  -> Could not extract a text snippet for this section. Skipping.")

    print("\n--- Summary Generation Complete ---")
    return refined_results