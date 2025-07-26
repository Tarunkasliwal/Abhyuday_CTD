import fitz
import pandas as pd
import numpy as np
import pickle
import json
import os
import tempfile
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, TFAutoModel

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Constants & Global Variables ---
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 32
MODEL = None
TOKENIZER = None
SCALER = None
LABEL_ENCODER = None
NUM_FEATURE_COLS = None

# --- Helper Functions ---
def get_casing(text):
    if text.isupper() and len(text) > 1: return 'upper'
    if text.istitle() and len(text) > 1: return 'title'
    return 'lower'

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_bbox = fitz.Rect(line["bbox"])
                    line_center = (line_bbox.x0 + line_bbox.x1) / 2
                    alignment = 'centered' if abs(line_center - page_width / 2) < (page_width * 0.15) else 'left'
                    for span in line["spans"]:
                        text = span['text'].strip()
                        if text:
                            data.append({
                                "text": text, "font_size": round(span['size'], 2),
                                "bold": 1 if "bold" in span['font'].lower() else 0,
                                "alignment": alignment, "x": round(span['bbox'][0], 2),
                                "y": round(span['bbox'][1], 2), "casing": get_casing(text),
                                "page": page_num + 1
                            })
    doc.close()
    df = pd.DataFrame(data)
    if not df.empty:
        df['y_position_normalized'] = df.groupby('page')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        page_font_mode = df.groupby('page')['font_size'].transform(lambda x: x.mode()[0])
        df['relative_font_size'] = df['font_size'] / page_font_mode
    return df

def find_document_title(df):
    if df.empty: return "", -1
    first_page_df = df[df['page'] == 1]
    if first_page_df.empty: return "", -1
    max_font_size = first_page_df['font_size'].max()
    potential_titles = first_page_df[first_page_df['font_size'] == max_font_size]
    if potential_titles.empty: return "", -1
    title_row = potential_titles.loc[potential_titles['y'].idxmin()]
    return title_row['text'], title_row.name

def predict_with_model(df, model, tokenizer, scaler, label_encoder, num_feature_cols):
    if df.empty: return df
    
    input_ids, attention_masks = [], []
    for text in df['text']:
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    df_encoded = pd.get_dummies(df, columns=['alignment', 'casing'], drop_first=True)
    for col in num_feature_cols:
        if col not in df_encoded.columns: df_encoded[col] = 0
    X_numerical = scaler.transform(df_encoded[num_feature_cols])

    predictions = model.predict({
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_masks),
        'numerical_input': X_numerical
    })
    predicted_indices = np.argmax(predictions, axis=1)
    df['predicted_label'] = label_encoder.inverse_transform(predicted_indices)
    return df

def load_all_models():
    """Loads all model components into global variables."""
    global MODEL, TOKENIZER, SCALER, LABEL_ENCODER, NUM_FEATURE_COLS
    print("Loading all models...")
    MODEL = load_model('english_distilbert_model.h5', custom_objects={"TFAutoModel": TFAutoModel})
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    with open('english_label_encoder.pkl', 'rb') as f: LABEL_ENCODER = pickle.load(f)
    with open('english_scaler.pkl', 'rb') as f: SCALER = pickle.load(f)
    with open('english_numerical_features.json', 'r') as f: NUM_FEATURE_COLS = json.load(f)
    print("Models loaded successfully.")

@app.route("/extract_outline", methods=["POST"])
def extract_outline():
    """API endpoint to extract the document outline from a PDF."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.save(tmp.name)
                tmp_pdf_path = tmp.name

            results_df = extract_features_from_pdf(tmp_pdf_path)
            title_text, title_index = find_document_title(results_df)
            results_df = predict_with_model(results_df, MODEL, TOKENIZER, SCALER, LABEL_ENCODER, NUM_FEATURE_COLS)

            final_json = {"title": title_text, "outline": []}
            if title_index != -1:
                results_df = results_df.drop(title_index, errors='ignore')

            headings_df = results_df[results_df['predicted_label'] != 'Body'].copy()
            for _, row in headings_df.iterrows():
                level_int = int(row['predicted_label'].replace('H', '')) if row['predicted_label'].startswith('H') else 0
                
                final_json["outline"].append({
                    "heading": row['text'],
                    "level": level_int,
                    "page": int(row['page'])
                })

            return jsonify(final_json)

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        finally:
            if 'tmp_pdf_path' in locals() and os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
    else:
        return jsonify({"error": "Invalid file type, please upload a PDF"}), 400

if __name__ == '__main__':
    load_all_models()
    app.run(host='0.0.0.0', port=5000)