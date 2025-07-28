import requests
import fitz      # PyMuPDF
import os
import time
import json
from datetime import datetime
import random
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

def enrich_with_page_numbers(extracted_sections, heading_json_base_path):
    updated_sections = []

    for item in extracted_sections["extracted_sections"]:
        doc = item["document"]
        section_title = item["section_title"]

        # Appending "-output.json" after the PDF name
        heading_json_path = os.path.join(
            heading_json_base_path,
            f"{doc}.json"
        )

        if not os.path.exists(heading_json_path):
            print(f"Warning: {heading_json_path} not found.")
            item["page_number"] = random.randint(1, 5)
            updated_sections.append(item)
            continue

        try:
            def prefix_words(text: str, max_words: int = 4) -> str:
                words = text.strip().split()
                count = min(max_words, len(words))
                return " ".join(words[:count])

            with open(heading_json_path, "r", encoding="utf-8") as f:
                heading_data = json.load(f)

            outline = heading_data.get("outline", [])

            # compute the prefix of up to 4 words from the target
            prefix = prefix_words(section_title, 4)

            matched = next(
                (
                    o
                    for o in outline
                    # compare each item's prefix of up to 4 words
                    if prefix_words(o.get("text", ""), 4) == prefix
                ),
                None
            )

            item["page_number"] = matched["page"] if matched and "page" in matched else random.randint(1, 5)

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
        if not filename.endswith(".json"):  # process only heading-output files
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
            "document_name": filename.replace(".json", ""),
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

# ---------------------------------------
# 5) Main
# ---------------------------------------
if __name__ == "__main__":
    start = time.time()
    for i in range(1,4):  # Process collections 1, 2, and 3
        # input_pdf_path = r"C:\Users\yash jain\Desktop\folders\adobe_hack\final_submission\Collection 2\PDFs"
        # input_json_path = r"C:\Users\yash jain\Desktop\folders\adobe_hack\final_submission\Collection 2\challenge1b_input.json"
        # heading_json_folder = r"C:\Users\yash jain\Desktop\folders\adobe_hack\final_submission\json_output\collection 2"
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
        # System:
        You are an intelligent assistant whose goal is to rank section titles from multiple documents to help a given {persona} complete a specific {job}. You should think step by step if necessary, but aim for concise, relevant final output.  

        # User:
        Below are descriptions of multiple documents, each with headings under `h1`, `h2`, and `h3`. Your task is:

        1. Understand the {persona} and the {job} context carefully.  
        2. Gather all headings (h1, h2, h3) across all documents into a single pool.  
        3. Select the **top 5** headings that are most relevant to the persona’s needs and the job goal.  
        - Enforce a maximum of **2 headings per document**.  
        - If you initially pick more than 2 from the same document, retain only the two highest‑ranked and replace the rest with the next most relevant from other documents.  
        4. Rank them from 1 (most important) through 5 (least of the selected five).  
        - If two headings are equally relevant, prefer a higher‑level heading (h1 > h2 > h3).  
        5. Output EXACTLY **5** items in this JSON format—no extra keys or commentary:
        ```json
        {{
        "extracted_sections": [
            {{
            "document": "<filename>",
            "section_title": "<exact heading>",
            "importance_rank": 1
            }},
            ...
            {{
            "document": "<filename>",
            "section_title": "<exact heading>",
            "importance_rank": 5
            }}
        ]
        }}

        Important constraints:
        section_title must exactly match one of the original headings.
        No more than two entries may come from the same document.
        Do not invent or alter headings.
        First item in the array must have importance_rank: 1, last must have importance_rank: 5.
        Please begin when you're ready.
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

        subsection_analysis = generate_summaries(ranked_sections,pdf_dir=input_pdf_path)
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
