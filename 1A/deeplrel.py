import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.utils import to_categorical
# ** UPDATED IMPORTS **
from transformers import DistilBertTokenizer, TFDistilBertModel

# --- 1. Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = pd.read_csv('1A\data\heading_dataset3_corrected.csv')
df.dropna(subset=['label', 'text'], inplace=True)
df['text'] = df['text'].astype(str)

# Feature Engineering
df['y_position_normalized'] = df.groupby('page')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
page_font_mode = df.groupby('page')['font_size'].transform(lambda x: x.mode()[0])
df['relative_font_size'] = df['font_size'] / page_font_mode

# Encode Labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
y = to_categorical(df['label_encoded'], num_classes=num_classes)

# --- Prepare Text Data with DistilBERT Tokenizer ---
# ** UPDATED MODEL NAME **
MODEL_NAME = 'distilbert-base-multilingual-cased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 32

def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True, return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

X_input_ids, X_attention_masks = bert_encode(df['text'], tokenizer)

# --- Prepare Numerical Data ---
df_encoded = pd.get_dummies(df, columns=['alignment', 'casing'], drop_first=True)
numerical_features = [
    'font_size', 'bold', 'x', 'y', 'y_position_normalized', 'relative_font_size'
]
for col in df_encoded.columns:
    if 'alignment_' in col or 'casing_' in col:
        numerical_features.append(col)

scaler = StandardScaler()
X_numerical = scaler.fit_transform(df_encoded[numerical_features])

# --- Split Data ---
X_train_ids, X_test_ids, X_train_masks, X_test_masks, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_input_ids, X_attention_masks, X_numerical, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. Build the Hybrid Model ---
print("Building the multilingual DistilBERT model...")
# ** UPDATED BERT MODEL CLASS **
distilbert_model = TFDistilBertModel.from_pretrained(MODEL_NAME)
distilbert_model.trainable = False

input_ids = Input(shape=(MAX_LEN,), dtype='int32', name='input_ids')
attention_mask = Input(shape=(MAX_LEN,), dtype='int32', name='attention_mask')
bert_output = distilbert_model(input_ids, attention_mask=attention_mask)[0]
cls_token = bert_output[:, 0, :]

numerical_input = Input(shape=(X_train_num.shape[1],), name='numerical_input')
dense_num = Dense(64, activation='relu')(numerical_input)

concatenated = concatenate([cls_token, dense_num])
dropout = Dense(128, activation='relu')(concatenated)
output = Dense(num_classes, activation='softmax')(dropout)

model = Model(inputs=[input_ids, attention_mask, numerical_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Train the Model ---
print("\nTraining the model...")
model.fit(
    {'input_ids': X_train_ids, 'attention_mask': X_train_masks, 'numerical_input': X_train_num},
    y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1
)

# --- 4. Evaluate and Save ---
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(
    {'input_ids': X_test_ids, 'attention_mask': X_test_masks, 'numerical_input': X_test_num}, y_test
)
print(f"\nFinal Model Accuracy on Test Set: {accuracy:.4f}")

print("Saving model and supporting files...")
model.save('distilbert_heading_model.h5') # New name for the smaller model
with open('label_encoder.pkl', 'wb') as f: pickle.dump(label_encoder, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('numerical_features.json', 'w') as f: json.dump(numerical_features, f)
print("All files saved successfully.")