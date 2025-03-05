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
model = load_model("chatbot_model.keras")  # Ensure filename matches

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

# Chatbot loop
print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    intent = predict_intent(user_input)
    response = get_response(intent)
    print(f"Chatbot: {response}")
