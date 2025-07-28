
import os
import json
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("models/en_core_web_sm/en_core_web_sm-3.8.0")


def load_outline_jsons(output_dir):
    outlines = []
    for fname in os.listdir(output_dir):
        if fname.endswith('.json'):
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                data['source_doc'] = fname.replace('.json', '.pdf')
                outlines.append(data)
    return outlines
def extract_section_context_by_outline(outlines, doc_name, section_index, input_dir="input"):
    this_entry = outlines[section_index]
    next_entry = outlines[section_index + 1] if section_index + 1 < len(outlines) else None

    start_page = this_entry["page"]
    pdf_path = os.path.join(input_dir, doc_name)
    doc = fitz.open(pdf_path)
    max_page = len(doc)

    end_page = next_entry["page"] if next_entry and next_entry["page"] > start_page else start_page + 1
    end_page = min(end_page, max_page)

    text = ""
    for page_num in range(start_page - 1, end_page):
        text += doc[page_num].get_text()

    return text.strip()


def noun_phrase_overlap(task_text: str, candidate_text: str) -> float:
    """Compute Jaccard overlap between noun phrases in task and section."""
    task_doc = nlp(task_text.lower())
    candidate_doc = nlp(candidate_text.lower())

    task_nouns = set(chunk.text.strip().lower() for chunk in task_doc.noun_chunks)
    cand_nouns = set(chunk.text.strip().lower() for chunk in candidate_doc.noun_chunks)

    if not task_nouns:
        return 0.0
    return len(task_nouns & cand_nouns) / len(task_nouns | cand_nouns)

def get_section_candidates(outlines):
    candidates = []
    for doc in outlines:
        for i, entry in enumerate(doc.get("outline", [])):
            candidates.append({
                "document": doc.get("source_doc", ""),
                "page": entry["page"],
                "section_title": entry["text"],
                "outline": doc.get("outline", []),
                "section_index": i
            })
    return candidates


def encode_all(model, texts, batch_size=16):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False)
def build_universal_query(persona: str, job_to_be_done: str) -> str:
    """
    Constructs a descriptive, domain-neutral query prompt for semantic search
    that can be used with any persona and job-to-be-done on any document collection.

    Args:
        persona (str): The role or expertise of the user.
        job_to_be_done (str): The concrete task to be accomplished.

    Returns:
        str: A dynamically generated query prompt to guide semantic ranking.
    """
    query = (
        f"You are acting as a {persona}, and your task is to: {job_to_be_done}. "
        "From the collection of document sections provided, identify and select "
        "those sections that contain the most relevant, specific, and actionable information "
        "to effectively complete the given job. "
        "Prioritize content that uniquely contributes toward solving the task, "
        "while avoiding generic, structural, or irrelevant sections such as introductions, indexes, or lists of contents. "
        "For each selected section, provide the document name, page number, section title, "
        "and a concise summary explaining its relevance to the persona’s objective."
    )
    return query
def extract_named_entities(text):
    doc = nlp(text)
    # Use entity text normalized to lowercase
    entities = set(ent.text.lower() for ent in doc.ents)
    return entities

def named_entity_overlap(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def select_relevant_sections(
        outlines,
        persona,
        job_to_be_done,
        input_dir="input",
        top_k=7):
    model = SentenceTransformer(os.path.join("models", "all-MiniLM-L6-v2"))

    candidates = get_section_candidates(outlines)
    query = build_universal_query(persona, job_to_be_done)
    print(query)
    job_focus_query = (
    f"What content would be most useful for a {persona} to successfully {job_to_be_done}? "
    "Prioritize sections that are specific, actionable, and useful for executing the task."
)
    job_focus_emb = model.encode([job_focus_query])[0]



  

 
    

    query_emb = model.encode([query])[0]

    title_texts = [c["section_title"] for c in candidates]
    context_texts = [
    extract_section_context_by_outline(c["outline"], c["document"], c["section_index"], input_dir)
    for c in candidates
]



    title_embs = encode_all(model, title_texts)
    combined_text = [c["section_title"] + "\n" + context_texts[i] for i, c in enumerate(candidates)]
    context_embs = encode_all(model, combined_text)
    job_entities = extract_named_entities(job_to_be_done)
    

    for i, cand in enumerate(candidates):
        title_sim = cosine_similarity([query_emb], [title_embs[i]])[0][0]
        context_sim = cosine_similarity([query_emb], [context_embs[i]])[0][0]
        noun_overlap = noun_phrase_overlap(query, context_texts[i])
        usefulness_sim = cosine_similarity([job_focus_emb], [context_embs[i]])[0][0]

        cand_entities = extract_named_entities(combined_text[i])
        entity_score = named_entity_overlap(job_entities, cand_entities)
        cand["score"] = (
    0.35 * context_sim + 
    0.35 * usefulness_sim+       
    0.15 * title_sim +         
    0.10 * noun_overlap +       
    0.10 * entity_score         
)


    
        cand["context"] = context_texts[i]


    candidates_sorted = sorted(candidates, key=lambda x: -x["score"])

    selected = []
    used = set()
    for cand in candidates_sorted:
        key = (cand["document"], cand["section_title"])
        if key in used:
            continue
        used.add(key)
        selected.append(cand)
        if len(selected) >= top_k:
            break

    return selected
