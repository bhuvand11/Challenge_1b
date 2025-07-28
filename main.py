import os, json
from outline_extractor import extract_outline_from_pdf
from section_selector import select_relevant_sections
import joblib
from datetime import datetime

MODEL_PATH = "models/heading_model_ensemble.joblib"
rf, gb = joblib.load(MODEL_PATH)

input_dir = "input"
output_dir = "output"
input_json = "challenge1b_input.json"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
doc_filenames = [doc["filename"] for doc in data["documents"]]

persona = " ".join(str(v) for v in data.get("persona", {}).values())
job_to_be_done = " ".join(str(v) for v in data.get("job_to_be_done", {}).values())

outlines = []
for docfile in doc_filenames:
    pdf_path = os.path.join(input_dir, docfile)
    outline = extract_outline_from_pdf(pdf_path, rf, gb)
    outline["source_doc"] = docfile
    outlines.append(outline)

selected_sections = select_relevant_sections(outlines, persona, job_to_be_done, input_dir=input_dir, top_k=7)

result = {
    "metadata": {
        "challenge_info": data.get("challenge_info", {}),
        "documents": doc_filenames,
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "processing_timestamp": datetime.now().isoformat()
    },
    "sections": [
        {
            "document": s["document"],
            "page_number": s["page"],
            "section_title": s["section_title"],
            "importance_rank": idx + 1
        }
        for idx, s in enumerate(selected_sections)
    ],
    "sub_section_analysis": [
        {
            "document": s["document"],
            "page_number": s["page"],
            "section_title": s["section_title"],
            "refined_text": s["context"]
        }
        for s in selected_sections
    ]
}

output_path = os.path.join(output_dir, "challenge1b_output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Output written to {output_path}")
