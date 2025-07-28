import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict
import json
import re
import os
import glob
from to_chunk_json import pdf_to_json_chunks

def process_pdf_to_json(pdf_path, output_path):
    """
    Process a PDF file and generate a structured JSON output with title and outline.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str): Path where the output JSON will be saved
    
    Returns:
        dict: The final schema output containing title and outline
    """
    
    def extract_lines_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        all_lines = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block_num, block in enumerate(blocks):
                if "lines" in block:
                    for line_num, line in enumerate(block["lines"]):
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        all_lines.append({
                            "page": page_num,
                            "block_number": block_num,
                            "line_number": line_num,
                            "bbox": line["bbox"],
                            "text": line_text.strip(),
                            "spans":line["spans"]
                        })
        return all_lines

    def generate_height_class_json_merged(lines_data, bins=10):
        heights = [(line["bbox"][3] - line["bbox"][1]) for line in lines_data]
        counts, bin_edges = np.histogram(heights, bins=bins)

        max_bin_index = np.argmax(counts)
        max_bin_upper_edge = bin_edges[max_bin_index + 1]

        # Only consider non-zero bins above the most common one
        valid_bins = []
        for i in range(len(counts)):
            if counts[i] > 0 and bin_edges[i + 1] > max_bin_upper_edge:
                valid_bins.append((i, (bin_edges[i], bin_edges[i + 1])))

        valid_bins.sort(key=lambda x: x[1][1], reverse=True)
        bin_class_map = {index: f"h{rank + 1}" for rank, (index, _) in enumerate(valid_bins)}

        grouped_blocks = defaultdict(list)

        for line in lines_data:
            height = line["bbox"][3] - line["bbox"][1]
            if height <= max_bin_upper_edge:
                continue

            bin_index = np.digitize(height, bin_edges, right=False) - 1
            if bin_index == len(bin_edges) - 1:
                bin_index -= 1

            if bin_index not in bin_class_map:
                continue

            key = (line["page"], line["block_number"])
            grouped_blocks[key].append((line["text"], height, bin_class_map[bin_index], line["bbox"], line["spans"]))

        output = []
        for (page, block_number), lines in grouped_blocks.items():
            merged_text = " ".join([t[0] for t in lines])
            height_class = sorted(set(t[2] for t in lines), key=lambda h: int(h[1:]))[0]
            top_line = min(lines, key=lambda x: x[3][1])  # get line with topmost bbox
            output.append({
                "text": merged_text.strip(),
                "block_number": block_number,
                "page": page,
                "height_class": height_class,
                "bbox": lines[0][3],  # any bbox
                "spans": sum((t[4] for t in lines), [])  # flatten list of spans
            })

        return output

    def is_table_like(block, span_threshold=2, line_threshold=2):
        """
        Heuristic to detect if a block is likely to be part of a table.
        span_threshold: number of spans per line to be considered columnar
        line_threshold: number of lines required to consider it tabular
        """
        if "lines" not in block or len(block["lines"]) < line_threshold:
            return False

        columnar_lines = 0
        for line in block["lines"]:
            if len(line["spans"]) >= span_threshold:
                columnar_lines += 1

        return columnar_lines >= line_threshold

    def filter_out_table_blocks(pdf_path, merged_json, span_threshold=2, line_threshold=2):
        doc = fitz.open(pdf_path)
        table_keys = set()
        table_zones = defaultdict(list)  # page → list of (y0, y1) of known tables

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block_num, block in enumerate(blocks):
                if "lines" not in block:
                    continue
                if is_table_like(block, span_threshold, line_threshold):
                    table_keys.add((page_num, block_num))
                    y0 = min(line["bbox"][1] for line in block["lines"])
                    y1 = max(line["bbox"][3] for line in block["lines"])
                    table_zones[page_num].append((y0, y1))

        # Optional: merge overlapping table zones per page
        def overlaps(a, b):
            return not (a[1] < b[0] or b[1] < a[0])

        def merge_ranges(ranges):
            if not ranges:
                return []
            ranges.sort()
            merged = [ranges[0]]
            for current in ranges[1:]:
                last = merged[-1]
                if overlaps(last, current):
                    merged[-1] = (min(last[0], current[0]), max(last[1], current[1]))
                else:
                    merged.append(current)
            return merged

        for page in table_zones:
            table_zones[page] = merge_ranges(table_zones[page])

        # Filter blocks that either are tables or fall into table zones
        filtered = []
        for block in merged_json:
            page = block["page"]
            block_id = block["block_number"]
            bbox = block.get("bbox", None)
            y0 = bbox[1] if bbox else None
            y1 = bbox[3] if bbox else None

            in_table_range = any(
                (y0 is not None and y1 is not None and y0 < zone[1] and y1 > zone[0])
                for zone in table_zones.get(page, [])
            )

            if (page, block_id) not in table_keys and not in_table_range:
                filtered.append(block)

        return filtered

    def normalize_text(text):
        return text.lower().strip().replace(" ", "").replace("\n", "")

    def remove_repeated_headers_footers(blocks, total_pages, min_repeat_ratio=0.5, y_threshold=100):
        """
        Removes all instances of text that appears on multiple pages AND typically resides
        near the top or bottom (but not solely based on position).
        """
        if total_pages <= 1:
            print("Skipping header/footer removal: only one page found.")
            return blocks

        content_occurrences = defaultdict(set)  # normalized_text → set of pages
        text_positions = defaultdict(list)      # normalized_text → list of Y positions

        # Step 1: Track text usage across pages
        for block in blocks:
            page = block["page"]
            text = normalize_text(block["text"])
            y0 = block.get("bbox", [0, 0, 0, 0])[1]
            y1 = block.get("bbox", [0, 0, 0, 0])[3]

            content_occurrences[text].add(page)
            text_positions[text].append((y0, y1))

        # Step 2: Identify repeated content with likely header/footer position
        repeated_candidates = set()
        for text, pages in content_occurrences.items():
            repeat_ratio = len(pages) / total_pages
            if repeat_ratio >= min_repeat_ratio:
                # Check if majority of this text's positions are near top or bottom
                y_hits = text_positions[text]
                top_hits = sum(1 for y0, _ in y_hits if y0 < y_threshold)
                bottom_hits = sum(1 for _, y1 in y_hits if y1 > 792 - y_threshold)
                if top_hits + bottom_hits >= len(y_hits) * 0.7:  # 70% near top/bottom
                    repeated_candidates.add(text)

        print(f"Identified {len(repeated_candidates)} repeated header/footer elements.")

        # Step 3: Filter out ALL blocks that match repeating text
        filtered = [
            block for block in blocks
            if normalize_text(block["text"]) not in repeated_candidates
        ]

        return filtered

    def is_block_bold(block):
        spans = block.get("spans", [])
        for span in spans:
            font_name = span.get("font", "").lower()
            if "bold" in font_name.replace(" ", ""):
                return True
        return False

    def apply_title_logic(merged_height_json):
        total_pages = len(set(block["page"] for block in merged_height_json))
        print(f"[INFO] Total pages in PDF: {total_pages}")

        h1_blocks = [b for b in merged_height_json if b["height_class"] == "h1"]
        h2_blocks = [b for b in merged_height_json if b["height_class"] == "h2"]

        scenario_1 = all(b["page"] == 0 for b in h1_blocks)
        print(f"[INFO] Scenario 1 (all h1 on page 0): {scenario_1}")

        page_0_height = 792  # Default A4 height; override if needed
        cutoff = page_0_height * 0.6
        print(f"[INFO] 60% page height cutoff: {cutoff}")

        if scenario_1:
            if total_pages == 1:
                print("[INFO] Only one page found.")
                h1_top = [b for b in h1_blocks if b["page"] == 0 and b.get("bbox", [0, 0, 0, 0])[1] < cutoff]
                h2_top = [b for b in h2_blocks if b["page"] == 0 and b.get("bbox", [0, 0, 0, 0])[1] < cutoff]
                print(f"[INFO] h1_top blocks: {len(h1_top)}, h2_top blocks: {len(h2_top)}")

                if h1_top and not h2_top:
                    topmost = min(h1_top, key=lambda b: b["bbox"][1])
                    topmost["height_class"] = "title"
                    print(f"[TITLE SET] From h1 only: '{topmost['text'][:60]}'")

                elif h1_top and h2_top:
                    top_h1 = min(h1_top, key=lambda b: b["bbox"][1])
                    top_h2 = min(h2_top, key=lambda b: b["bbox"][1])
                    print(f"[INFO] top_h1 y={top_h1['bbox'][1]}, text='{top_h1['text'][:50]}'")
                    print(f"[INFO] top_h2 y={top_h2['bbox'][1]}, text='{top_h2['text'][:50]}'")

                    if top_h1["bbox"][1] < top_h2["bbox"][1]:
                        top_h1["height_class"] = "title"
                        print(f"[TITLE SET] From higher h1: '{top_h1['text'][:60]}'")
                    else:
                        if is_block_bold(top_h2):
                            top_h2["height_class"] = "title"
                            print(f"[TITLE SET] From bold h2: '{top_h2['text'][:60]}'")
                        else:
                            top_h1["height_class"] = "title"
                            print(f"[TITLE SET] Default to h1: '{top_h1['text'][:60]}'")

                elif not h1_top and h2_top:
                    candidate = min(h2_top, key=lambda b: b["bbox"][1])
                    if is_block_bold(candidate):
                        candidate["height_class"] = "title"
                        print(f"[TITLE SET] From bold h2: '{candidate['text'][:60]}'")
                    else:
                        print("[INFO] No valid h1/h2 for title → inserting dummy")
                        dummy_title = {
                            "text": "",
                            "height_class": "title",
                            "block_number": -1,
                            "page": 0
                        }
                        merged_height_json.insert(0, dummy_title)

                else:
                    print("[INFO] No top h1 or h2 found → inserting dummy")
                    dummy_title = {
                        "text": "",
                        "height_class": "title",
                        "block_number": -1,
                        "page": 0
                    }
                    merged_height_json.insert(0, dummy_title)

            else:
                print("[INFO] Multiple pages — promoting h1 on page 0 if exists")
                for block in merged_height_json:
                    if block["height_class"].startswith("h"):
                        current_rank = int(block["height_class"][1:])
                        if current_rank == 1 and block["page"] == 0:
                            if not any(b["height_class"] == "title" for b in merged_height_json):
                                topmost = min([b for b in h1_blocks if b["page"] == 0], key=lambda b: b["bbox"][1])
                                topmost["height_class"] = "title"
                                print(f"[TITLE SET] Topmost h1 on page 0: '{topmost['text'][:60]}'")
                        elif current_rank >= 2:
                            block["height_class"] = f"h{current_rank - 1}"
        else:
            print("[INFO] h1 blocks are on multiple pages")
            h1_page0 = [b for b in h1_blocks if b["page"] == 0]
            if h1_page0:
                topmost = min(h1_page0, key=lambda b: b["bbox"][1])
                topmost["height_class"] = "title"
                print(f"[TITLE SET] From page 0 h1: '{topmost['text'][:60]}'")

        return merged_height_json

    def clean_final_json(data):
        cleaned = []
        for block in data:
            text = block.get("text", "").strip()

            # Skip if:
            if (
                len(text) <= 1 or                             # single char
                re.fullmatch(r"\d+", text) or                # only digits
                re.fullmatch(r"[^\w\s]", text)               # single punctuation/symbol
            ):
                continue

            cleaned.append({
                "text": text,
                "block_number": block.get("block_number", -1),
                "height_class": block.get("height_class", ""),
                "page": block.get("page", 0)
            })

        return cleaned

    def custom_filter_entries(data):
        """
        Applies two different filtering strategies:
        - If height_class is 'title' and text has more than 15 words → keep entry but set text to ""
        - If height_class is 'h1' or any other (not 'title') and text has more than 20 words → remove the entry
        """
        filtered = []

        for item in data:
            word_count = len(item["text"].split())
            height = item.get("height_class", "")

            if height == "title":
                if word_count > 17:
                    # Keep the entry but blank out the text
                    item = item.copy()
                    item["text"] = ""
                filtered.append(item)
            else:
                if word_count <= 21:
                    filtered.append(item)

        return filtered

    def map_output_to_chunks(output_list, chunks_list):
        """
        Walk through output_list (json1) and chunks_list (json2) in tandem,
        finding for each output_list[i] the first chunks_list[j] (j monotonic)
        such that *every* token in output_list[i]['text'] appears (as a
        substring, case‐insensitive) in chunks_list[j]. When found, replaces
        output_list[i]['text'] with chunks_list[j] and moves on (i++ but
        keep j anchored).
        """
        mapped = []
        j = 0

        for item in output_list:
            text1 = item['text']
            # split on whitespace, drop empties
            tokens = [tok for tok in re.split(r'\s+', text1) if tok]
            matched = False

            # scan chunks from current j onward
            for k in range(j, len(chunks_list)):
                chunk = chunks_list[k]
                low_chunk = chunk.lower()
                if all(tok.lower() in low_chunk for tok in tokens):
                    # got a match
                    new_item = item.copy()
                    new_item['text'] = chunk
                    mapped.append(new_item)
                    j = k        # anchor j here for the next round
                    matched = True
                    break

            if not matched:
                # no chunk contained all tokens: leave text as-is
                mapped.append(item.copy())

        return mapped

    def translate_to_schema_format(mapped_data):
        """
        Translates the mapped_new structure to the required schema format
        with title and outline structure.
        """
        # Find the title (entry with height_class == "title")
        title_entry = None
        outline_entries = []
        
        for item in mapped_data:
            if item.get("height_class") == "title":
                title_entry = item
            else:
                # Convert height_class to level format
                height_class = item.get("height_class", "")
                level = "H1" if height_class == "h1" else "H2" if height_class == "h2" else "H3"
                
                outline_entries.append({
                    "level": level,
                    "text": item.get("text", ""),
                    "page": item.get("page", 1)
                })
        
        # Create the final schema structure
        schema_output = {
            "title": title_entry.get("text", "") if title_entry else "",
            "outline": outline_entries
        }
        
        return schema_output

    # Main processing pipeline
    bins = 10
    
    # Extract lines from PDF
    lines_data = extract_lines_from_pdf(pdf_path)
    
    # Generate height-based classification
    merged_height_json = generate_height_class_json_merged(lines_data, bins=bins)
    
    # Apply title logic
    new_json_with_title = apply_title_logic(merged_height_json)
    
    # Step 1: Remove table blocks
    filtered_no_tables = filter_out_table_blocks(pdf_path, new_json_with_title)
    
    # Step 2: Remove repeated headers/footers
    total_pages = len(set(line["page"] for line in lines_data))
    final_filtered = remove_repeated_headers_footers(filtered_no_tables, total_pages)
    
    # Clean the final JSON
    final_output = clean_final_json(final_filtered)
    
    # Get chunks for mapping
    chunks = pdf_to_json_chunks(pdf_path)
    
    # Map output to chunks
    mapped = map_output_to_chunks(final_output, chunks)
    mapped_new = custom_filter_entries(mapped)
    
    # Update page numbers by adding 1 to each page
    for item in mapped_new:
        item["page"] = item["page"] + 1
    
    # Translate to schema format
    final_schema_output = translate_to_schema_format(mapped_new)
    
    # Save to output file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_schema_output, f, indent=2, ensure_ascii=False)
    
    print(f"Processed PDF and saved output to: {output_path}")
    
    return final_schema_output

def process_pdfs_directory(input_directory, output_directory):
    """
    Process all PDF files in the input directory and save JSON outputs to the output directory.
    
    Args:
        input_directory (str): Path to the directory containing PDF files
        output_directory (str): Path to the directory where JSON outputs will be saved
    
    Returns:
        list: List of processed file paths and their corresponding output paths
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_pattern = os.path.join(input_directory, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {input_directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    processed_files = []
    
    for pdf_file in pdf_files:
        try:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]
            
            # Create output JSON filename
            output_json = os.path.join(output_directory, f"{base_name}.json")
            
            print(f"\nProcessing: {os.path.basename(pdf_file)}")
            print(f"Output will be saved to: {output_json}")
            
            # Process the PDF
            result = process_pdf_to_json(pdf_file, output_json)
            
            processed_files.append({
                "input_pdf": pdf_file,
                "output_json": output_json,
                "result": result
            })
            
            print(f"✓ Successfully processed: {os.path.basename(pdf_file)}")
            
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(pdf_file)}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Successfully processed {len(processed_files)} out of {len(pdf_files)} files.")
    return processed_files

# Docker execution
if __name__ == "__main__":
    # Docker paths
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    print(f"Processing PDFs from: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Process all PDFs in the input directory
    results = process_pdfs_directory(input_dir, output_dir)
    
    if results:
        print(f"\nSuccessfully processed {len(results)} PDF files:")
        for result in results:
            print(f"  - {os.path.basename(result['input_pdf'])} -> {os.path.basename(result['output_json'])}")
    else:
        print("No PDF files were processed.")
