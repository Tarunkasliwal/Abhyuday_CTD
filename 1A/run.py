import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import pickle
import json
import os
import glob
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

# --- Constants: Define paths for models and directories ---
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 32
INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'
MODEL_PATH = 'multilingual_heading_model_v2.h5'
LABEL_ENCODER_PATH = 'label_encoder_v2.pkl'
SCALER_PATH = 'scaler_v2.pkl'
FEATURES_PATH = 'numerical_features_v2.json'

def get_casing(text):
    """Determines the casing of a string (upper, title, or lower)."""
    if text.isupper() and len(text) > 1:
        return 'upper'
    if text.istitle() and len(text) > 1:
        return 'title'
    return 'lower'

def extract_features_from_pdf(pdf_path):
    """Extracts text and layout features from each span in a PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
    
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
                    # Determine text alignment based on its position
                    alignment = 'centered' if abs(line_center - page_width / 2) < (page_width * 0.15) else 'left'
                    for span in line["spans"]:
                        text = span['text'].strip()
                        if text:
                            data.append({
                                "text": text,
                                "font_size": round(span['size'], 2),
                                "bold": 1 if "bold" in span['font'].lower() else 0,
                                "alignment": alignment,
                                "x": round(span['bbox'][0], 2),
                                "y": round(span['bbox'][1], 2),
                                "casing": get_casing(text),
                                "page": page_num + 1
                            })
    doc.close()
    
    df = pd.DataFrame(data)
    if not df.empty:
        # Normalize y-position and font size relative to the page
        df['y_position_normalized'] = df.groupby('page')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        page_font_mode = df.groupby('page')['font_size'].transform(lambda x: x.mode()[0] if not x.mode().empty else x.mean())
        df['relative_font_size'] = df['font_size'] / (page_font_mode + 1e-6)
    return df

def find_document_title(df):
    """Identifies the document title, assuming it's the highest text with the largest font on page 1."""
    if df.empty: return "", -1
    first_page_df = df[df['page'] == 1]
    if first_page_df.empty: return "", -1
    
    max_font_size = first_page_df['font_size'].max()
    potential_titles = first_page_df[first_page_df['font_size'] == max_font_size]
    if potential_titles.empty: return "", -1
    
    title_row = potential_titles.loc[potential_titles['y'].idxmin()]
    return title_row['text'], title_row.name

def predict_with_multilingual_model(df, model, tokenizer, scaler, label_encoder, num_feature_cols):
    """Runs the ML model to predict heading labels for the extracted text features."""
    if df.empty: return df

    # Tokenize text for BERT model
    input_ids, attention_masks = [], []
    for text in df['text']:
        encoded = tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    # Prepare numerical and categorical features
    df_encoded = pd.get_dummies(df, columns=['alignment', 'casing'], drop_first=True)
    for col in num_feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded.reindex(columns=num_feature_cols, fill_value=0)
    X_numerical = scaler.transform(df_encoded[num_feature_cols])

    # Make predictions
    predictions = model.predict({
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_masks),
        'numerical_input': X_numerical
    })
    predicted_indices = np.argmax(predictions, axis=1)
    df['predicted_label'] = label_encoder.inverse_transform(predicted_indices)
    return df

def process_single_pdf(pdf_path, model, tokenizer, scaler, label_encoder, num_feature_cols):
    """Orchestrates the processing of a single PDF file."""
    print(f"Processing: {os.path.basename(pdf_path)}")
    results_df = extract_features_from_pdf(pdf_path)
    if results_df is None or results_df.empty:
        print(f"Could not extract any text from {pdf_path}. Skipping.")
        return {"title": "Extraction Failed", "outline": []}

    title_text, title_index = find_document_title(results_df)
    results_df = predict_with_multilingual_model(results_df, model, tokenizer, scaler, label_encoder, num_feature_cols)

    # Format the final JSON output
    final_json = {"title": title_text, "outline": []}
    if title_index != -1:
        results_df = results_df.drop(title_index, errors='ignore')

    headings_df = results_df[results_df['predicted_label'] != 'Body'].copy()
    headings_df = headings_df.sort_values(by=['page', 'y'])

    for _, row in headings_df.iterrows():
        final_json["outline"].append({
            "level": row['predicted_label'],
            "text": row['text'],
            "page": int(row['page'])
        })
    return final_json

def main():
    """Main entry point: loads models and processes all PDFs in the input directory."""
    print("--- Starting Document Processing ---")
    try:
        model = load_model(MODEL_PATH, custom_objects={"TFBertModel": TFBertModel})
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        with open(LABEL_ENCODER_PATH, 'rb') as f: label_encoder = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        with open(FEATURES_PATH, 'r') as f: num_feature_cols = json.load(f)
    except Exception as e:
        print(f"FATAL: Error loading required model files: {e}. Exiting.")
        return

    pdf_files = glob.glob(os.path.join(INPUT_DIR, '*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    for pdf_path in pdf_files:
        try:
            base_name = os.path.basename(pdf_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{file_name_without_ext}.json")
            
            json_output = process_single_pdf(pdf_path, model, tokenizer, scaler, label_encoder, num_feature_cols)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved output to {output_path}")
        except Exception as e:
            print(f"An error occurred while processing {pdf_path}: {e}")

    print("--- Document Processing Complete ---")

if __name__ == '__main__':
    main()
