
import pickle  # For loading the trained model and tokenizer
import numpy as np  # For handling probabilities

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
import emoji
import re
from nltk.tokenize import word_tokenize
from langdetect import detect
import nltk
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
import contractions

from flask import Flask, render_template, request ,jsonify



import os

# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths for your files
model_path = os.path.join(base_dir, "lstm2m.h5")
tokenizer_path = os.path.join(base_dir, "tokenizer10k.pkl")

# Load model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)


textc = "she was in dilemma when i saw her yestarday"

stopwords = stopwords.words('english')
swdr_neg = set(stopwords) - set(['no','nor','not','never','against'])

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def remove_num(comm):
    return re.sub(r"[0-9]", "", comm).strip()

def remove_tags(comm):
    return re.sub(r'[@#]\S+', '', comm).strip()

def remove_splch(comm):
    return re.sub(r'[^A-Za-z0-9\s]', ' ', comm)



def remove_stp(txt):
    st=""
    for val in list(txt.split(" ")):
        if val not in swdr_neg:
            st=st+val+" "
    return st

lemmatizer = WordNetLemmatizer()
def preprcs_text(text):
    
    tokens = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_words)


def preprocess_pipeline(text):
    text = contractions.fix(text)
    text = remove_emojis(text)
    text = remove_num(text)
    text = remove_tags(text)
    text = remove_splch(text)
    text = remove_stp(text)
    text = preprcs_text(text)
    return text




# Initialize the Flask app
app = Flask(__name__)



# Map labels to emotions
label_to_emotion = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'joy',
    4: 'neutral',
    5: 'sadness',
    6: 'surprise'
}

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for emotion prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get('text')  # Get text input from the request
    

    #Applying preprocessing
    input_text=preprocess_pipeline(input_text)


    #creatngg vector for input
    seq= tokenizer.texts_to_sequences([input_text])

    padded_comm = pad_sequences(seq, maxlen=50, padding='post')

    print(padded_comm)


    probabilities = model.predict(padded_comm)[0]





    # Prepare data for response: a list of emotion-probability pairs
    emotion_data = [
        {'emotion': label_to_emotion[i], 'probability': round(prob * 100, 2)}
        for i, prob in enumerate(probabilities)
    ]

    print(f"Received input text: {input_text}")
    # print(f"Vectorized input: {input_vector}")
    print(f"Predicted probabilities: {probabilities}")
    print(f"Emotion data sent to frontend: {emotion_data}")

    return jsonify({'emotion_data': emotion_data}) 

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
