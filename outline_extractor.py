import fitz 
import re
import pandas as pd

import os, json, joblib
from collections import defaultdict
import numpy as np 
MODEL_PATH = 'models/heading_model_ensemble.joblib'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
def extract_features_from_span(span, page_num, all_spans_on_page):
    text = span["text"].strip()
    cur_size = span.get("size", 0)
    cur_font = span.get("font", "")
    left, top, right, bottom = span.get("bbox", [0,0,0,0])
    sizes = [s.get("size", 0) for s in all_spans_on_page] or [cur_size]
    rel_size_rank = (cur_size - min(sizes)) / (max(sizes)-min(sizes)+1e-3)

    page_width = max([s.get('bbox', [0,0,0,0])[2] for s in all_spans_on_page] or [612])
    center_x = (left + right) / 2
    align_score = abs(center_x - page_width / 2) / page_width
    is_centered = int(align_score < 0.15)
    cap_ratio = sum([c.isupper() for c in text])/(len(text)+1e-2)
    tokens = text.strip().split(' ')
    is_numbered = int(tokens[0].replace('.','').isdigit()) if tokens else 0
    sorted_y = sorted([s.get("bbox",[0,0,0,0])[1] for s in all_spans_on_page])
    y_idx = sorted_y.index(top) if top in sorted_y else -1
    spacing_before = top-sorted_y[y_idx-1] if y_idx>0 else 999
    spacing_after = (sorted_y[y_idx+1]-bottom) if y_idx+1<len(sorted_y) else 999
    surrounding_text = ''.join([s.get("text","") for s in all_spans_on_page[max(0,y_idx-1):y_idx+2]])
    is_isolated = int(len(surrounding_text) < 2*len(text))

    return [
        round(cur_size, 2),                      
        int(bool(span.get("flags", 0) & 2)),   
        int(text.istitle()),                      
        int(text.isupper()),                     
        page_num,                                 
        int(text[:1].isdigit()) if text else 0,  
        len(text),                                
        len(text.split()),                        
        rel_size_rank,                            
        is_numbered,                              
        cap_ratio,                              
        align_score,                                      
        is_centered,                              
        spacing_before,                          
        spacing_after,                            
        is_isolated   
                                                          
    ]
def text_spans_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pdf_spans = []
    page_widths = {}
    for page_num, page in enumerate(doc, 1):
        page_spans = []
        
        page_widths[page_num] = page.rect.width 
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text","").strip()
                    if text:
                        page_spans.append(span)
                        pdf_spans.append((span, page_num, text))
    return pdf_spans,page_widths

def starts_with_number(text):
    return bool(re.match(r"^\d+\.\s+", text))

def is_num_heading(text):
    t = text.replace('\u00a0', ' ')
    return bool(re.match(r"^\d+(\.\d+)*\s+", t))




def is_valid_heading(text):
    alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    return alnum_ratio > 0.4 


def compute_running_headers(spans):
    y_text_map = defaultdict(list)
    pages = set(pg for _, pg, _ in spans)
    N = len(pages)
    for span, page_num, text in spans:
        y = span.get('bbox', [0,0,0,0])[1]
        if page_num >= 3 and y < 100:
            continue
        y_text_map[round(y,1)].append((page_num, text))
    running_texts = set()
    for y, items in y_text_map.items():
        count = len(items)
        if count > 0.3 * N:
            for _, text in items:
                running_texts.add(text.strip())
    return running_texts


def merge_multiline_headings(spans, x_threshold=3, y_threshold=3):
    spans = sorted(spans, key=lambda s: (s[1], s[0].get("bbox")[1], s[0].get("bbox")[0]))
    merged_spans = []
    i = 0
    while i < len(spans):
        span, pg, txt = spans[i]
        cur_y = span.get('bbox')[1]
        cur_page = pg
        cur_line = [(span, pg, txt)]
        j = i + 1
        while j < len(spans):
            next_span, next_pg, next_txt = spans[j]
            if next_pg != cur_page:
                break
            next_y = next_span.get('bbox')[1]
            if abs(next_y - cur_y) > y_threshold:
                break
            cur_line.append((next_span, next_pg, next_txt))
            j += 1
        cur_line = sorted(cur_line, key=lambda s: s[0].get("bbox")[0])
        prev_right = None
        merged_text = ""
        for l_span, _, l_txt in cur_line:
            x_left = l_span.get("bbox")[0]
            if prev_right is not None and x_left - prev_right > x_threshold:
                merged_text += " "
            merged_text += l_txt.strip()
            prev_right = l_span.get("bbox")[2]
        merged_spans.append((cur_line[0][0], cur_page, merged_text))
        i = j
    return merged_spans
def ensemble_predict_with_proba(rf, gb, X):
    rf_probs = rf.predict_proba(X)
    gb_probs = gb.predict_proba(X)
    classes = rf.classes_
    preds = []
    for i in range(len(X)):
        rf_label = rf.predict(X.iloc[i].values.reshape(1, -1))[0]
        gb_label = gb.predict(X.iloc[i].values.reshape(1, -1))[0]

        if rf_label == gb_label:
            preds.append(rf_label)
        else:
            rf_heading_conf = sum([rf_probs[i][np.where(classes == c)[0][0]] for c in ["Title","H1","H2","H3"] if c in classes])
            gb_heading_conf = sum([gb_probs[i][np.where(classes == c)[0][0]] for c in ["Title","H1","H2","H3"] if c in classes])
            rf_body_conf = rf_probs[i][np.where(classes == "body")[0][0]] if "body" in classes else 0
            gb_body_conf = gb_probs[i][np.where(classes == "body")[0][0]] if "body" in classes else 0

            if (rf_heading_conf > rf_body_conf) or (gb_heading_conf > gb_body_conf):
                if rf_heading_conf > gb_heading_conf:
                    preds.append(rf_label)
                else:
                    preds.append(gb_label)
            else:
                preds.append("body")
    return preds


def merge_line_spans(line_spans, x_threshold):
    line_spans = sorted(line_spans, key=lambda s: s[0].get('bbox')[0])  
    merged_text = line_spans[0][2]
    prev_right = line_spans[0][0].get('bbox')[2]

    span, page, _ = line_spans[0]

    for cur_span, cur_page, cur_text in line_spans[1:]:
        cur_left = cur_span.get('bbox')[0]
        gap = cur_left - prev_right
        if gap > x_threshold:
            merged_text += ' ' + cur_text
        else:
            merged_text += cur_text  
        prev_right = cur_span.get('bbox')[2]

    return (span, page, merged_text)


def join_multiline_headings(outline):
    result = []
    i = 0
    while i < len(outline):
        cur = outline[i]
        to_merge = [cur["text"]]
        while (i+1 < len(outline) and 
               outline[i]["level"] == outline[i+1]["level"] and 
               abs(outline[i]["y"] - outline[i+1]["y"]) < 25):

            next_text = outline[i+1]["text"].strip()
            if re.match(r'^(\d+(\.\d+)*\.?)\s+', next_text):
                break
            if re.match(r'^[A-Z][a-z]', next_text):
                break

            to_merge.append(next_text)
            i += 1

        new_entry = {k: v for k, v in cur.items() if k != "text"}
        new_entry["text"] = " ".join(to_merge)
        result.append(new_entry)
        i += 1
    return result
def get_center_distance(span, page_num, page_widths):
    bbox = span.get('bbox', [0, 0, 0, 0])
    left, right = bbox[0], bbox[2]
    center_x = (left + right) / 2
    pw = page_widths.get(page_num, 612)
    return abs(center_x - pw / 2)




def robust_title_from_spans(title_lines, title_spans, title_pages, all_headings, running_headers, page_widths, ensemble_pred):
    good_titles = [
    (txt, span, pg) for (txt, span, pg), label in zip(zip(title_lines, title_spans, title_pages), ensemble_pred)
    if pg in (1, 2)
    and label in ("Title", "H1")
    and len(txt.strip()) > 10
    and txt.strip().lower() not in running_headers
]


    for txt, span, pg in good_titles:
        _ = get_center_distance(span, pg, page_widths)  

    good_titles = sorted(
        good_titles,
        key=lambda x: (get_center_distance(x[1], x[2], page_widths), x[1].get('bbox', [0, 0, 0, 0])[1])
    )

    seen, merged = set(), []
    for t, _, _ in good_titles:
        t = t.strip()
        if t and t not in seen:
            merged.append(t)
            seen.add(t)
    result = " ".join(merged).strip()
    if result:
        return result

    best_heading = None
    best_size = -1
    for h in all_headings:
        if h['page'] in (1, 2):
            if h['text'].strip().lower() in running_headers:
                continue
            if h['text'].strip().lower() in ('contents', 'table of contents'):
                continue
            size = h.get('size', 0)
            y = h.get('y', 0)
            candidate_score = (size, -y, len(h['text']))
            if candidate_score > (best_size, 0, 0):
                best_heading = h
                best_size = size
    if best_heading:
        return best_heading['text'].strip()

    if all_headings:
        return all_headings[0]['text'].strip()

    return ""  

def merge_number_with_heading(spans):
    merged_spans = []
    i = 0
    while i < len(spans):
        text = spans[i][2].strip()
        if re.match(r'^\d+\.$', text) and (i+1) < len(spans):
            next_span, next_pg, next_text = spans[i+1]
            combined_text = f"{text} {next_text.strip()}"
            combined_span = spans[i][0].copy()  
            merged_spans.append((combined_span, next_pg, combined_text))
            i += 2
        else:
            merged_spans.append(spans[i])
            i += 1
   
    
    return merged_spans
def clean_outline(outline):
    cleaned = []
    for entry in outline:
        text = entry['text'].strip()
        
        if len(text.split()) > 10:
            continue

        if text.endswith('.'):
            continue

        if text and text[0].islower():
            continue

        if ',' in text and len(text.split()) > 6:
            continue

        if len(text) < 5 and not text.isupper():
            continue

        cleaned.append(entry)
    return cleaned


def extract_outline_from_pdf(pdf_path, rf, gb):
    spans, page_widths = text_spans_from_pdf(pdf_path)

    spans = merge_number_with_heading(spans)
    spans = merge_multiline_headings(spans)

    page2spans = defaultdict(list)
    for s, pg, _ in spans:
        page2spans[pg].append(s)
    running_headers = compute_running_headers(spans)
    title_lines, title_spans, title_pages = [], [], []
    outline = []

    features_batch = []
    texts_batch = []
    for i, (span, page_num, text) in enumerate(spans):
        if text in running_headers and page_num not in (1, 2):
            continue
        feats = extract_features_from_span(span, page_num, page2spans[page_num])
        features_batch.append(feats)
        texts_batch.append((span, page_num, text))

    if not features_batch:
        return {"title": "", "outline": []}
    feature_columns = [
        "font_size", "bold", "is_title", "is_caps", "page", "starts_with_number",
        "length", "num_words", "rel_font_size", "is_num_heading", "cap_ratio",
        "align_score", "is_centered", "spacing_before", "spacing_after", "is_isolated"
    ]
       

   



    X_batch_df = pd.DataFrame(features_batch, columns=feature_columns)
    ensemble_pred = ensemble_predict_with_proba(rf, gb, X_batch_df)

  


  

    for (span, page_num, text), label in zip(texts_batch, ensemble_pred):
        norm_text = text.strip().lower()
        
    
      
        bbox = span.get("bbox", [0, 0, 0, 0])
        y = bbox[1]
        font_size = span.get("size", 0)
        bold = bool(span.get("flags", 0) & 2)
        avg_font_size = np.mean([s.get("size", 0) for s in page2spans[page_num]])

        def get_heading_level(text, font_size, avg_font_size, y):
            if len(text.strip().split()) > 12:
                return None
            if text.endswith('.'):
                return None
            if ',' in text and len(text.split()) > 6:
                return None
            if text.strip()[0:1].islower():
                return None
            if font_size < 8:
                return None

            match = re.match(r"^(\d+(\.\d+)*)\.?\s+", text)
            if match:
                depth = len(match.group(1).split("."))
                if depth == 1:
                    return "H1"
                elif depth == 2:
                    return "H2"
                else:
                    return "H3"

            if font_size > avg_font_size * 1.1 and y < 150:
                return "H1"
            elif font_size > avg_font_size * 0.95 and y < 250:
                return "H2"
            return None

        def get_heading_level_later_pages(text, font_size, avg_font_size, y, bold):
            text = text.strip()

            if font_size < 8:
                return None
            if len(text) > 120 or len(text.split()) > 12:
                return None
            if text.endswith('.'):
                return None
            if ',' in text and len(text.split()) > 6:
                return None
            if text[0:1].islower():
                return None
            if text.startswith("• "):  
                return None
            if len(text) < 5 and not text.isupper():
                return None

            if bold and y < 150 and font_size >= avg_font_size * 0.9:
                return "H1"
            if y < 250 and font_size >= avg_font_size * 0.8:
                return "H2"
            return None


        if label == "body":
            if page_num in (1, 2):
                new_label = get_heading_level(text, font_size, avg_font_size, y)
            else:
                new_label = get_heading_level_later_pages(text, font_size, avg_font_size, y, bold)
            if new_label:
                label = new_label

        if label == "Title":
            title_lines.append(text)
            title_spans.append(span)
            title_pages.append(page_num)
        elif label in ("H1", "H2", "H3") and is_valid_heading(text):
            outline.append({"level": label, "text": text, "page": page_num, "y": y})

    outline = join_multiline_headings(outline)
    outline = clean_outline(outline)

    title = robust_title_from_spans(title_lines, title_spans, title_pages, outline, running_headers, page_widths, ensemble_pred)
    if not title and outline:
        title = outline[0]["text"]
    def normalize_text(text):
        return re.sub(r'\s+', ' ', text.strip()).casefold()

    normalized_title = normalize_text(title)
    outline = [
    entry for entry in outline
    if normalize_text(entry["text"]) != normalized_title
]


    return {
    "title": title,
    "outline": [{ "level": item["level"], "text": item["text"], "page": item["page"] } for item in outline]
}


