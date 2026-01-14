# 🎬 IMDB Movie Review Sentiment Analysis

A comprehensive End-to-End Deep Learning project that classifies movie reviews as **Positive** or **Negative**. 

This project demonstrates the transition from raw text data to a deployed web application using **Long Short-Term Memory (LSTM)** networks and **Streamlit**.

## 🚀 Overview

Sentiment analysis is a crucial task in Natural Language Processing (NLP). This project utilizes the Keras IMDB dataset (containing 50,000 reviews) to train a Recurrent Neural Network. The trained model is then serialized and integrated into an interactive web interface where users can type their own reviews and get real-time feedback.

## 🛠️ Tech Stack

### Core Frameworks
* **Python 3.x**: Primary programming language.
* **TensorFlow / Keras**: Used for building and training the Deep Learning model.
* **Streamlit**: Used to create the web application frontend.

### Data Processing & Utilities
* **NumPy**: Matrix operations and data handling.
* **Matplotlib**: Visualization of training performance (Loss/Accuracy).
* **Pickle/H5**: Model serialization.

## 🧠 Model Architecture

The model solves the "Vanishing Gradient" problem found in SimpleRNNs by utilizing an **LSTM (Long Short-Term Memory)** architecture.

### Data Pipeline
1.  **Vocabulary Size:** Top 1,000 most frequent words.
2.  **Sequence Length:** Reviews padded/truncated to 200 words.
3.  **Word Indexing:** specific offset handling (Index + 3) to match IMDB standards (`<PAD>`, `<START>`, `<UNK>`, `<UNUSED>`).

### Network Structure
The network consists of three main layers:

1.  **Embedding Layer**: 
    * *Input:* Integer sequences (Batch Size, 200).
    * *Function:* Transforms integers into dense vectors of size 128.
    * *Output:* (Batch Size, 200, 128).
2.  **LSTM Layer**:
    * *Units:* 128 neurons.
    * *Activation:* Tanh (internal).
    * *Function:* Processes the sequence and maintains long-term context.
3.  **Dense (Output) Layer**:
    * *Units:* 1 neuron.
    * *Activation:* Sigmoid.
    * *Output:* Probability score between 0.0 (Negative) and 1.0 (Positive).

### Architecture Diagram
```mermaid
graph LR
    A[Input Text] --> B(Tokenization & Padding)
    B --> C[Embedding Layer<br/>(1000 -> 128 dims)]
    C --> D[LSTM Layer<br/>(128 Units)]
    D --> E[Dense Layer<br/>(Sigmoid)]
    E --> F[Output Probability]