import fitz
import pandas as pd
import numpy as np
import pickle
import json
import os
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

# Constants
MODEL_NAME = 'bert-base-multilingual-cased'
MAX_LEN = 32

def get_casing(text):
    if text.isupper() and len(text) > 1: return 'upper'
    if text.istitle() and len(text) > 1: return 'title'
    return 'lower'

def extract_features_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
    doc = fitz.open(pdf_path)
    data = []
    # (Extraction logic is the same as before)
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
        # ** NEW FEATURE CALCULATION **
        df['y_position_normalized'] = df.groupby('page')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        page_font_mode = df.groupby('page')['font_size'].transform(lambda x: x.mode()[0])
        df['relative_font_size'] = df['font_size'] / page_font_mode
    return df

def find_document_title(df):
    # (This function is the same as before)
    if df.empty: return "", -1
    first_page_df = df[df['page'] == 1]
    if first_page_df.empty: return "", -1
    max_font_size = first_page_df['font_size'].max()
    potential_titles = first_page_df[first_page_df['font_size'] == max_font_size]
    if potential_titles.empty: return "", -1
    title_row = potential_titles.loc[potential_titles['y'].idxmin()]
    return title_row['text'], title_row.name
    
def predict_with_multilingual_model(df, model, tokenizer, scaler, label_encoder, num_feature_cols):
    # (This function is the same as before)
    if df.empty: return df

    input_ids = []
    attention_masks = []
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

def main():
    input_pdf = input("Enter the path to your input PDF file: ")
    output_json = input("Enter the desired name for the output JSON file: ")
    
    print("\nLoading multilingual model (v2) and supporting files...")
    try:
        model = load_model('multilingual_heading_model_v2.h5', custom_objects={"TFBertModel": TFBertModel})
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        with open('label_encoder_v2.pkl', 'rb') as f: label_encoder = pickle.load(f)
        with open('scaler_v2.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('numerical_features_v2.json', 'r') as f: num_feature_cols = json.load(f)
    except Exception as e:
        print(f"Error loading required files: {e}. Please run the v2 training script first.")
        return

    # The rest of the main function is identical to the previous version
    print(f"Extracting features from '{input_pdf}'...")
    results_df = extract_features_from_pdf(input_pdf)
    if results_df is None or results_df.empty:
        print("Processing stopped.")
        return

    print("Identifying document title...")
    title_text, title_index = find_document_title(results_df)

    print("Predicting heading labels with the multilingual model...")
    results_df = predict_with_multilingual_model(results_df, model, tokenizer, scaler, label_encoder, num_feature_cols)

    final_json = {"title": title_text, "outline": []}
    if title_index != -1:
        results_df = results_df.drop(title_index, errors='ignore')

    headings_df = results_df[results_df['predicted_label'] != 'Body'].copy()
    for _, row in headings_df.iterrows():
        final_json["outline"].append({
            "level": row['predicted_label'], "text": row['text'], "page": int(row['page'])
        })

    print(f"Saving structured results to '{output_json}'...")
    with open(output_json, 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print("Process complete!")

if __name__ == '__main__':
    main()