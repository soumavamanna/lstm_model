import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# --- Configuration ---
MAX_LEN = 200
MAX_FEATURES = 1000

# --- Load Model ---
@st.cache_resource
def load_prediction_model():
    try:
        model = tf.keras.models.load_model('lstm_imdb.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Did you run the training script first?")
        return None

model = load_prediction_model()

# --- Preprocessing Helper ---
def preprocess_text(text):
    """
    Converts raw text string into a padded integer sequence
    compatible with the IMDB dataset mapping.
    """
    # 1. Get the word index from keras
    word_index = imdb.get_word_index()
    
    # 2. Adjust index by +3 (IMDB dataset convention: 0=PAD, 1=START, 2=UNK, 3=UNUSED)
    #    We create a reverse dictionary for lookup, ensuring we stick to the top 1000 words.
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    # 3. Tokenize
    # Remove punctuation and split by space (simple tokenization)
    tokens = text.lower().split()
    
    # 4. Map to integers
    int_sequence = []
    for word in tokens:
        # Get index, default to 2 (UNK) if word not found
        idx = word_index.get(word, 2) 
        
        # If index is >= 1000 (max_features), treat as UNK (2)
        if idx >= MAX_FEATURES:
            idx = 2
        int_sequence.append(idx)
            
    # 5. Pad sequence
    padded_sequence = sequence.pad_sequences([int_sequence], maxlen=MAX_LEN)
    return padded_sequence

# --- Streamlit UI ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to check if it's Positive or Negative using an **LSTM** model.")

user_input = st.text_area("Review Text:", "This movie was absolutely fantastic! The acting was great.")

if st.button("Analyze Sentiment"):
    if model and user_input:
        with st.spinner("Analyzing..."):
            # Preprocess
            processed_input = preprocess_text(user_input)
            
            # Predict
            prediction = model.predict(processed_input)
            score = prediction[0][0]
            
            # Display Result
            st.markdown("---")
            if score > 0.5:
                st.success(f"**Sentiment: Positive** (Confidence: {score:.2f})")
                st.balloons()
            else:
                st.error(f"**Sentiment: Negative** (Confidence: {score:.2f})")
    elif not user_input:
        st.warning("Please enter some text.")