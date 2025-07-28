# Challenge 1B — Approach Explanation

## 📘 Overview

This solution implements a **persona- and job-driven semantic search pipeline** to extract and rank the most relevant sections from arbitrary PDFs. It is designed to be fully domain-agnostic, modular, explainable, and capable of running offline via Docker in a Linux AMD64 environment.

---

## 🚀 Key Objectives

* Extract only the most **contextually relevant** sections of a document.
* Stay **domain-independent** — no hardcoded assumptions about document structure.
* Use a blend of **semantic and lexical signals** for ranking.
* Produce a clean, **structured JSON** output with metadata.

---

## 🧱 System Components

### 1. 🧭 Outline Extraction

* Uses PDF span features like font size, layout, and positioning to identify hierarchical **headings**.
* Outputs a structured outline with section titles, levels, and page numbers.

### 2. 📄 Context Extraction

* For each heading, extracts text starting from that heading up to the next heading or page break.
* These are treated as **section candidates** for semantic ranking.

### 3. 🛠️ Prompt Construction

* Builds a **dynamic query prompt** using the user's persona and job description.
* Encourages specificity and actionability in extracted results.

### 4. 🔍 Embedding and Encoding

* Concatenates each section’s heading + body, then embeds it using `sentence-transformers/all-MiniLM-L6-v2`.
* Both the user’s base query and job-focused prompt are also embedded.

### 5. 🧠 Multi-Signal Relevance Scoring

Each section is scored using a **weighted formula** combining five signals:

```python
0.35 * context_sim +
0.35 * usefulness_sim +
0.15 * title_sim +
0.10 * noun_overlap +
0.10 * entity_score
```

* `context_sim`: Similarity between query and section text
* `usefulness_sim`: Similarity between job-specific prompt and section
* `title_sim`: Similarity between query and section title
* `noun_overlap`: Jaccard overlap between query and section’s noun phrases
* `entity_score`: Jaccard overlap of named entities

### 6. 🚫 Filtering & Deduplication

* Filters out generic or boilerplate sections (e.g., "References", "Appendix").
* Ensures **no duplicate section titles** per document in final output.

### 7. 🧾 Output Format

* Ranks the final results and saves them as a clean JSON with fields:

  * `document`, `page_number`, `section_title`, `relevance_score`, and `section_text`

---

## ⚠️ Known Limitations

* Embeddings may confuse similar but unrelated terms.
* Section context is limited to headings/page boundaries.
* Occasional appearance of generic or partially relevant matches.
* No current mechanism to ensure **semantic diversity** among results.

---

## 🔧 Enhancement Strategies

* **Prompt tuning** to better condition embeddings.
* **Hybrid scoring** (semantic + lexical) for greater accuracy.
* **Heading filtering & deduplication** to suppress noise.
* Future additions:

  * Clustering for semantic diversity
  * Multi-page section modeling
  * Summarization or LLM-based reranking (if permitted)

---

## ✅ Summary

This approach combines robustness, relevance, and flexibility:

* 📄 Works on **any type of PDF**
* 🔗 Uses **combined semantic and linguistic matching**
* 🛠️ Is **modular, explainable, and production-ready**
* 🐳 Fully Dockerized and **offline-compatible**

It meets the challenge requirements for personalized, explainable, persona/job-driven document section extraction.

