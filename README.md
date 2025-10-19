# shadowfox
# Install dependencies
# pip install tensorflow nltk

import nltk
nltk.download('punkt')

import numpy as np
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from difflib import get_close_matches

# --- Data Preparation ---
corpus = """
Hello, how are you doing today? I hope everything is going well.
Today is a great day to learn about neural networks.
This keyboard will predict the next word and autocorrect typos.
"""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

corpus = clean_text(corpus)
tokens = word_tokenize(corpus)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for i in range(1, len(tokens)):
    n_gram_sequence = tokens[:i+1]
    seq = tokenizer.texts_to_sequences([" ".join(n_gram_sequence)])[0]
    input_sequences.append(seq)

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = np.eye(total_words)[y]

# --- Model Training ---
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

# --- Prediction Functions ---
def predict_next_word(model, tokenizer, text, max_len):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    pred = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(pred)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

def autocorrect(word, vocab, n=1):
    suggestions = get_close_matches(word, vocab, n=n, cutoff=0.7)
    return suggestions[0] if suggestions else word

# --- Interactive Keyboard ---
vocab = list(tokenizer.word_index.keys())

while True:
    input_text = input("\nEnter text: ")
    words = input_text.strip().split()
    if words:
        last_word = words[-1]
        corrected = autocorrect(last_word, vocab)
        words[-1] = corrected
        print(f"Autocorrected: {' '.join(words)}")
    context = ' '.join(words)
    predicted_word = predict_next_word(model, tokenizer, context, max_seq_len)
    print(f"Next word suggestion: {predicted_word}")

