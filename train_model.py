import numpy as np
import json
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

# Configuration
MAX_LEN = 1000  # Maximum length of JSON strings
VOCAB_SIZE = 10000  # Vocabulary size
EMBED_DIM = 64  # Embedding dimension
NUM_FILTERS = 128  # Number of CNN filters
FILTER_SIZE = 5  # CNN filter size
BATCH_SIZE = 32
EPOCHS = 2

def generate_data(num_samples=5000):
    """Generate training data with valid/invalid JSON samples"""
    valid, invalid = [], []
    
    # Valid JSON samples
    for _ in range(num_samples):
        # Simple objects
        valid.append(json.dumps({"id": np.random.randint(1000), "active": bool(np.random.randint(2))}))
        
        # Nested objects
        valid.append(json.dumps({"user": {"name": "test", "roles": ["admin", "user"]}}))
        
        # Arrays
        valid.append(json.dumps({"values": [1, 2, 3], "settings": {"debug": True}}))

    # Invalid JSON samples
    for _ in range(num_samples):
        # Missing quotes
        invalid.append('{name: "test", "age": 30}')
        # Trailing comma
        invalid.append('{"a": 1, "b": 2,}')
        # Unclosed brackets
        invalid.append('{"values": [1, 2, 3')
        # Wrong values
        invalid.append('{"debug": TRUE}')
    
    return valid, invalid

def build_model():
    """Create CNN model for JSON validation"""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        Conv1D(NUM_FILTERS, FILTER_SIZE, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def main():
    print("Generating training data...")
    valid, invalid = generate_data()
    texts = valid + invalid
    labels = [1]*len(valid) + [0]*len(invalid)  # 1=valid, 0=invalid

    print("Tokenizing texts...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    y = np.array(labels)

    print("Building model...")
    model = build_model()
    
    # Save the best model
    checkpoint = ModelCheckpoint('json_validator.h5', 
                               save_best_only=True,
                               monitor='val_accuracy',
                               mode='max')

    print("Training model...")
    history = model.fit(X, y,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_split=0.2,
                       callbacks=[checkpoint])

    # Save tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Training complete. Saved:")
    print("- json_validator.h5 (CNN model)")
    print("- tokenizer.pkl (Text tokenizer)")

if __name__ == '__main__':
    main()