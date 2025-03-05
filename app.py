import streamlit as st
import json
import random
import numpy as np
import nltk
import spacy
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

# Load intents.json
with open("intents.json", "r") as file:
    data = json.load(file)

# Load trained model
model = load_model("chatbot_model.keras")

# Load words and classes
with open("words.pkl", "rb") as f:
    words = pickle.load(f)

with open("classes.pkl", "rb") as f:
    classes = pickle.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Ensure input size matches training
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array(bag).reshape(1, -1)  # Fix input shape issue

def predict_intent(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(bow)[0]
    return classes[np.argmax(res)] if max(res) > 0.5 else "unknown"

def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# Streamlit UI
st.title("AI Chatbot 🤖")
st.write("Ask me anything!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input field
user_input = st.text_input("You:", "")

if user_input:
    intent = predict_intent(user_input)
    response = get_response(intent)

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display new message
    with st.chat_message("assistant"):
        st.write(response)
