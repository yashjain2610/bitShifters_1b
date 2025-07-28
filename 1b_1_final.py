import requests
import fitz      # PyMuPDF
import os
import time
import json
from datetime import datetime
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from main_pipeline_pdfs import process_pdfs_directory
from subsection_analysis import generate_summaries

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

def extract_clean_json(response):
    _, sep, rest = response.partition("</think>")
    if sep:
        clean_output = rest.strip()
    else:
        clean_output = response.strip()


    pattern = r"^```.*?\n(.*)\n```$"
    m = re.match(pattern, clean_output, flags=re.DOTALL)
    clean_output =  m.group(1) if m else clean_output
    print(clean_output)
    return json.loads(clean_output)


# --- Extract text from a single PDF page ---
def extract_text_from_page(pdf_path, page_number):
    """
    Extracts text from a specific page (1-based) of a PDF file.
    """
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > doc.page_count:
        return ""
    page = doc.load_page(page_number - 1)
    return page.get_text()
# ---------------------------------------
# 1) Extract text from PDF
# ---------------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def enrich_with_page_numbers(extracted_sections, heading_json_base_path = "/app/json_output"):
    updated_sections = []

    for item in extracted_sections["extracted_sections"]:
        doc = item["document"]
        section_title = item["section_title"]

        # Appending "-output.json" after the PDF name
        heading_json_path = os.path.join(
            heading_json_base_path,
            f"{doc}-heading-output.json"
        )

        if not os.path.exists(heading_json_path):
            print(f"Warning: {heading_json_path} not found.")
            item["page_number"] = -1
            updated_sections.append(item)
            continue

        try:
            with open(heading_json_path, "r", encoding="utf-8") as f:
                heading_data = json.load(f)

            outline = heading_data.get("outline", [])
            matched = next((o for o in outline if o.get("text", "").strip() == section_title.strip()), None)

            item["page_number"] = matched["page"] if matched and "page" in matched else -1

        except Exception as e:
            print(f"Error reading {heading_json_path}: {e}")
            item["page_number"] = -1

        updated_sections.append(item)

    return json.dumps(updated_sections, indent=2) , updated_sections

def read_input_info(input_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    doc_list = data["documents"]
    filename_list = []
    for doc in doc_list: 
        filename_list.append(doc["title"])
    return data["persona"]["role"], data["job_to_be_done"]["task"] , filename_list , data["challenge_info"]["challenge_id"]

# --- Read heading structure jsons from folder and convert to required format ---
def build_headings_from_folder(json_folder):
    headings = []
    for filename in os.listdir(json_folder):
        if not filename.endswith("-heading-output.json"):  # process only heading-output files
            continue
        full_path = os.path.join(json_folder, filename)
        with open(full_path, 'r') as f:
            data = json.load(f)

        # If JSON has an "outline" key, use that list of items
        outline = data.get("outline") if isinstance(data, dict) else None
        if outline is None:
            # Skip if no outline available
            continue

        # Collect headings by level
        h1_list = [item.get("text", "") for item in outline if item.get("level", "").upper() == "H1"]
        h2_list = [item.get("text", "") for item in outline if item.get("level", "").upper() == "H2"]
        h3_list = [item.get("text", "") for item in outline if item.get("level", "").upper() == "H3"]

        headings.append({
            "document_name": filename.replace("-heading-output.json", ""),
            "h1": h1_list,
            "h2": h2_list,
            "h3": h3_list
        })
    return headings

# ---------------------------------------
# 2) Generic Ollama query
# ---------------------------------------
def query_ollama(model_name, prompt, stream=False, images=None, json_mode=False):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream
    }
    if images is not None:
        payload["images"] = images

    # Enable JSON mode if requested
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(f"{OLLAMA_API_URL}/generate", json=payload)
    if not resp.ok:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return resp.json().get("response", "")

# ---------------------------------------
# 3) Generate the custom ranking prompt
# ---------------------------------------
def generate_ranking_prompt(persona, job_to_be_done):
    """
    Sends a meta‐prompt to Qwen to return
    a "ranking‐instructions" prompt for your specific job.
    """
    meta_prompt = f"""
You are a {persona['role']}.
Your task is: {job_to_be_done['task']}.

I will give you a JSON array called "headings" where each element has:
  - "document": filename
  - "h1": list of level‑1 headings
  - "h2": list of level‑2 headings
  - "h3": list of level‑3 headings

Please generate one concise **instruction** prompt that, when used with that "headings" JSON, will tell a model to:
  1. Identify the top 5 section titles (could be from h1/h2/h3) most relevant to the task.
  2. Rank them by importance (1 = most important to 5 = fifth).
  3. Output **only** a JSON array named "extracted_sections" where each item has:
     • "document": filename  
     • "section_title": the heading text  
     • "importance_rank": integer 1–5

Return just that single instruction prompt (no JSON, no explanation).
""".strip()
    return query_ollama("qwen3:0.6b", meta_prompt.strip())

def rank_headings_with_prompt(headings_json, ranking_prompt):
    # embed the JSON as a literal in your prompt
    full_prompt = (
        f"Here is the heading hierarchy JSON:\n\n{json.dumps(headings_json, indent=2)}\n\n"
        + ranking_prompt
    )
    return query_ollama("qwen3:0.6b", full_prompt)


def build_final_output_json(persona, job, filename_list, extracted_sections,subsection_analysis, challenge_id,output_path):
    output = {
        "metadata": {
            "input_documents": filename_list,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis  # This remains empty for now
    }

    output_path = output_path + f"/challenge1b_output.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    return output


# ---------------------------------------
# 4) Process folder of PDFs with the generated prompt
# ---------------------------------------
def process_pdf_folder(folder_path, ranking_prompt, model_name="qwen3:0.6b"):
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.pdf')
    ]
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    # build one big prompt with headers
    combined_text = ""
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        combined_text += f"\n\n--- Contents of {os.path.basename(pdf_path)} ---\n{text}"

    full_prompt = combined_text + "\n\n" + ranking_prompt
    result = query_ollama(model_name, full_prompt)
    print(result)

def get_ranked_sections(headings_json, ranking_prompt):
    full_prompt = f"Here is the heading hierarchy JSON:\n\n{json.dumps(headings_json, indent=2)}\n\n{ranking_prompt}"
    print(full_prompt)
    return query_ollama("qwen3:0.6b", full_prompt,json_mode=True)

def process_entry(entry, persona, job, model_name, pdf_base_path):
    doc_name = entry["document"]
    page_num = entry.get("page_number", 1)
    pdf_path = os.path.join(pdf_base_path, doc_name + ".pdf")

    raw_text = extract_text_from_page(pdf_path, page_num)
    if not raw_text:
        return None

    prompt = f"""
    You are a {persona}.
    Your task is: {job}.

    Below is the extracted raw text for the section "{entry['section_title']}" from {doc_name}.pdf (page {page_num}):

    {raw_text}

    Please refine and succinctly summarize this section into a cohesive paragraph, preserving key details and clarity.
    Return only the refined text (no JSON).
"""

    refined = query_ollama(model_name, prompt)

    _, sep, rest = refined.partition("</think>")
    refined = rest.strip() if sep else refined.strip()

    return {
        "document": doc_name,
        "refined_text": refined,
        "page_number": page_num
    }


def build_subsection_analysis(ranked_sections_json, persona, job,
                              model_name="qwen3:0.6b",
                              pdf_base_path=r"/app/Collection 1/PDFs",
                              max_workers=5):

    ranked_sections = json.loads(ranked_sections_json)
    analysis = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_entry, entry, persona, job,
                            model_name, pdf_base_path): entry
            for entry in ranked_sections
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                analysis.append(result)
    return analysis

# def build_subsection_analysis(ranked_sections_json, persona, job,
#                               model_name="qwen3:0.6b",
#                               pdf_base_path=r"/app/Collection 1/PDFs",
#                               max_workers=5):
#     ranked_sections = json.loads(ranked_sections_json)
#     analysis = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(process_entry, entry, persona, job, model_name, pdf_base_path): entry
#             for entry in ranked_sections
#         }

#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 analysis.append(result)

#     return analysis




# def build_subsection_analysis(ranked_sections, persona, job, model_name="qwen3:0.6b",pdf_base_path = "/app/Collection 1/PDFs"):
#     """
#     For each entry in ranked_sections (with keys document, section_title, importance_rank, page_number),
#     extracts the raw text of that page from the PDF, sends it to the model with a refinement prompt,
#     and returns a list of dicts with document, refined_text, and page_number.
#     """
#     ranked_sections = json.loads(ranked_sections)

#     analysis = []
#     for entry in ranked_sections:
#         doc_name = entry["document"]
#         page_num = entry.get("page_number", 1)
#         pdf_path = os.path.join(pdf_base_path, doc_name + ".pdf")

#         raw_text = extract_text_from_page(pdf_path, page_num)
#         if not raw_text:
#             continue

#         # Craft prompt to refine that section's raw text
#         prompt = f"""
#             You are a {persona}.
#             Your task is: {job}.

#             Below is the extracted raw text for the section \"{entry['section_title']}\" from {doc_name}.pdf (page {page_num}):

#             """ + raw_text + """

#             Please refine and succinctly summarize this section into a cohesive paragraph,
#             preserving key details and clarity.
#             Return only the refined text (no JSON).
#         """

#         refined = query_ollama(model_name, prompt)

#         _, sep, rest = refined.partition("</think>")
#         if sep:
#             refined = rest.strip()
#         else:
#             refined = refined.strip()

#         analysis.append({
#             "document": doc_name,
#             "refined_text": refined.strip(),
#             "page_number": page_num
#         })
#     return analysis


# ---------------------------------------
# 5) Main
# ---------------------------------------
if __name__ == "__main__":
    start = time.time()
    for i in range(1, 4):  # Process collections 1, 2, and 3
        input_pdf_path = f"/app/Collection {i}/PDFs"
        input_json_path = f"/app/Collection {i}/challenge1b_input.json"
        heading_json_folder = f"/app/json_output/collection {i}"

        os.makedirs(heading_json_folder, exist_ok=True)
        #process pdfs to get 1a output
        process_pdfs_directory(input_pdf_path, heading_json_folder)
        # 1) Describe what the docs are about (this is optional/contextual)
        # with open("heading_hierarchy.json") as f:
        #     headings = json.load(f)

        # 3) Generate the custom ranking instruction prompt
        # ranking_prompt = generate_ranking_prompt(persona, job_to_be_done)
        persona, job , filename_list , challenge_id = read_input_info(input_json_path)
        headings = build_headings_from_folder(heading_json_folder)


        ranking_prompt = f"""
            ## Main Goal
            You are an intelligent assistant helping with ranking the headings of multiple documents on the basis of a given persona and job.

            Your objective is to **identify and rank the top 5 most relevant section titles** (from document outlines) that best align with the given **{persona}** and **{job}**.

            ---

            ## Step-by-Step Plan
            1. Read the `persona` and `task` descriptions carefully to understand the context and intent.
            2. Review all the provided headings of all the documents:
            - Each element represents a document with:
                - `document_name`: the source filename
                - `h1`: list of level‑1 section titles
                - `h2`: list of level‑2 section titles
                - `h3`: list of level‑3 section titles
            3. **Combine every heading** from every document into one pool.*Do not* implicitly weight Document 1 heavier just because it appears first.
            4. Choose the **top 5 section titles** that are **most relevant to completing the task**, regardless of their level.
            5. Rank them by importance:
                - `1` is the most useful or essential section.
                - `5` is the least important among the top five.
            6. Include which document the section was taken from.

            ---

            ## Guidelines
            - Prioritize clarity, relevance to the task, and coverage of diverse aspects and variety of documents.
            - striclty first review the all the headings before preaparing ranking
            - dont give more than 2 headings from same document to cover all the documents.
            - Avoid generic or repeated section titles unless they're crucial.
            - Do not infer or hallucinate extra information beyond the section titles.
            - If multiple titles are equally good, prefer higher-level (h1 > h2 > h3).
            - section_title should match **exactly** word to word with one of the headings from h1, h2, or h3.
            - you cant use document_filename for section title only h1 h2 or h3

            ---

            ## Strict Output JSON Format
            Return **only** a JSON array named `extracted_sections` of size 5, where each item is an object with:
            - `document`: the filename (from the `headings` JSON (document_name))
            - `section_title`: the selected heading (must match one from h1, h2, or h3)
            - `importance_rank`: integer between 1 and 5 (no duplicates)

            ### Example Output (follow this strictly):
            ```json
            [
            {{
                "document": "Example.pdf",
                "section_title": "Some Relevant Heading",
                "importance_rank": 1
            }},
            {{
                "document": "Example2.pdf",
                "section_title": "Another Heading",
                "importance_rank": 2
            }}
            // up to 5 items only
            ]
    """
        # print("=== Instruction Prompt ===")
        # print(ranking_prompt)
        # print()

        #process_pdf_folder(folder_path=r"C:\Users\yash jain\Desktop\folders\adobe_hack\Adobe-India-Hackathon25\Challenge_1b\Collection 1\PDFs", ranking_prompt=ranking_prompt)
        ranked_sections = get_ranked_sections(headings, ranking_prompt)

        ranked_sections = extract_clean_json(ranked_sections)
        print()
        print(ranked_sections)
        ranked_sections_json , ranked_sections = enrich_with_page_numbers(ranked_sections,heading_json_folder)

        print("\n--- Extracted Sections ---\n")
        print(ranked_sections)

        subsection_analysis = generate_summaries(ranked_sections_json,pdf_dir=input_pdf_path)
        print("\n--- Subsection Analysis ---\n")
        print(subsection_analysis)

        output_path = f"/app/Collection {i}"

        result = build_final_output_json(persona, job, filename_list, ranked_sections, subsection_analysis,challenge_id,output_path)
        print(result)

        end = time.time()
        print(f"Time taken: {end - start} seconds")
        # 4) Apply it to your headings JSON
        # extracted = rank_headings_with_prompt(headings, ranking_prompt)
        # print("=== Extracted Sections ===")
        # print(extracted)