# Challenge 1B — Relevant Section Extraction from PDFs


---

## 🌐 Overview

This repository contains a Dockerized, offline-compatible solution for **Adobe India Hackathon Challenge 1B**, which involves extracting only the most relevant sections of a PDF in response to a provided query.

All components run fully offline using pretrained models and local inference, and the setup is compatible with **Linux AMD64** environments via Docker.

---

## ✍️ Approach

### 1. Input Query

* The system begins with a query defined in `challenge1b_input.json`, which specifies what information to extract from the PDF.

### 2. Outline Extraction

* `outline_extractor.py` scans the PDF using `PyMuPDF` and extracts hierarchical sections based on layout, indentation, and font.

### 3. Semantic Matching

* `section_selector.py` embeds both the query and all extracted sections using the offline `sentence-transformers/all-MiniLM-L6-v2` model.
* Cosine similarity is used to find the top-k relevant matches.

### 4. Linguistic Enhancement

* `spaCy` is used for additional NLP preprocessing with the `en_core_web_sm` model, which is stored and linked offline.

### 5. Execution Pipeline

* The complete process is initiated by `main.py`, which loads the query, processes the PDF, selects relevant sections, and writes the final output JSON.

---

## 📦 Libraries Used

* `PyMuPDF`
* `pandas`, `numpy`
* `sentence-transformers`
* `scikit-learn`
* `spaCy`
* `joblib`

---

## 🤖 Models Used

* Sentence Embedding: `sentence-transformers/all-MiniLM-L6-v2`
* NLP Toolkit: `spaCy` with `en_core_web_sm` (manually linked inside Docker)
* Heading Detection: `models/heading_model_ensemble.joblib` (included for reuse or debugging — currently not invoked by default)

---

## 📂 Folder Structure

```
Challenge_1b/
├── main.py
├── section_selector.py
├── outline_extractor.py
├── challenge1b_input.json
├── input/                  # Input PDFs
│   └── sample.pdf
├── output/                 # Output JSON with relevant sections
│   └── .keep               # Placeholder for Git tracking
├── models/
│   ├── all-MiniLM-L6-v2/   # Sentence transformer model
│   ├── en_core_web_sm/     # Local spaCy model (v3.8.0)
│   └── heading_model_ensemble.joblib
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🐳 Docker Instructions

### 🔧 Build the Docker Image

```bash
docker build --platform linux/amd64 -t challenge1b-app .
```

### 🚀 Run the Container

```bash
docker run --rm \
  -v "$(pwd)/input":/app/input:ro \
  -v "$(pwd)/output":/app/output \
  --network none \
  challenge1b-app
```

---

## ✅ Output Format

* The result will be saved as a `.json` file in the `/output` folder.
* This JSON contains only the top-k sections semantically aligned to the query.

---

## 📎 Notes

* Runs **fully offline**
* Compatible with **Linux AMD64**
* Pretrained models stored in `/models` and loaded directly in Docker
* spaCy model is linked manually using:

  ```dockerfile
  RUN python -m spacy link models/en_core_web_sm/en_core_web_sm-3.8.0 en_core_web_sm
  ```

