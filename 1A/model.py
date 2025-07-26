
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.keras.utils import to_categorical

# --- 1. Load and Preprocess the Data ---

# Load the dataset
try:
    df = pd.read_csv('1A\data\heading_dataset3_corrected.csv')
except FileNotFoundError:
    print("Error: 'heading_dataset3_corrected.csv' not found.")
    exit()

# Drop rows with missing labels and text
df.dropna(subset=['label', 'text'], inplace=True)
df['text'] = df['text'].astype(str)

# --- Feature Engineering ---
# Add a feature for vertical position relative to the page
df['y_position_normalized'] = df.groupby('page')['y'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

# --- Encode Labels ---
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
y = to_categorical(df['label_encoded'], num_classes=num_classes)

# --- Prepare Text Data ---
# Tokenize the text (convert words to numbers)
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(df['text'])
X_text = tokenizer.texts_to_sequences(df['text'])
X_text = pad_sequences(X_text, maxlen=20) # Pad sequences to a fixed length

# --- Prepare Numerical and Categorical Data ---
# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['alignment', 'casing'], drop_first=True)

# Define numerical features
numerical_features = ['font_size', 'bold', 'x', 'y', 'y_position_normalized']
# Add the new dummy columns
for col in df_encoded.columns:
    if 'alignment_' in col or 'casing_' in col:
        numerical_features.append(col)

# Scale the numerical features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(df_encoded[numerical_features])

# --- Split the Data ---
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_numerical, y, test_size=0.2, random_state=42, stratify=y
)

print("Data preprocessed and split successfully.")

# --- 2. Build the Deep Learning Model ---

# Text input branch
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(text_input)
lstm_layer = LSTM(128)(embedding_layer)

# Numerical input branch
numerical_input = Input(shape=(X_train_num.shape[1],), name='numerical_input')
dense_layer_num = Dense(64, activation='relu')(numerical_input)

# Combine the branches
concatenated = concatenate([lstm_layer, dense_layer_num])
dense_1 = Dense(128, activation='relu')(concatenated)
output = Dense(num_classes, activation='softmax')(dense_1)

# Create the model
model = Model(inputs=[text_input, numerical_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 3. Train the Model ---
print("\nTraining the deep learning model...")
history = model.fit(
    [X_train_text, X_train_num],
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
print("Model training complete.")

# --- 4. Evaluate the Model ---
loss, accuracy = model.evaluate([X_test_text, X_test_num], y_test)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

# --- 5. Save the Model and Supporting Files ---
print("Saving model and supporting files...")
model.save('heading_detection_dl_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# Save the list of numerical feature columns
with open('numerical_features.json', 'w') as f:
    json.dump(numerical_features, f)

print("All files saved successfully.")