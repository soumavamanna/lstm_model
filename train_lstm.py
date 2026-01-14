import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.callbacks import EarlyStopping

# Data Loading & Preprocessing
max_features = 1000  # Only consider the top 1000 words
max_len = 200        # Cut off texts after 200 words

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Concatenating and re-splitting to match your specific 80/20 logic
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

split_index = int(0.8 * len(X))

X_train_final = X[:split_index]
y_train_final = y[:split_index]
X_test_final = X[split_index:]
y_test_final = y[split_index:]

# Pad sequences
X_train_final = sequence.pad_sequences(X_train_final, maxlen=max_len)
X_test_final = sequence.pad_sequences(X_test_final, maxlen=max_len)

print(f"Training shape: {X_train_final.shape}")

# Building LSTM Model 
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128)) 
model.add(LSTM(128, activation='tanh')) 
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

#  Train Model 
earlystopping = EarlyStopping(
    monitor='val_loss',
    patience=3, 
    restore_best_weights=True
)

history = model.fit(
    X_train_final,
    y_train_final,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

# Evaluate & Save 
test_loss, test_accuracy = model.evaluate(X_test_final, y_test_final)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save("lstm_imdb.h5")
print("Model saved as 'lstm_imdb.h5'")