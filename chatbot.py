from flask import Flask, render_template, request
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

FILE_PATH = 'All_Info.txt'
//comment
f = open(FILE_PATH, 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

sentence_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

[sentence_tokens[:2], word_tokens[:2]]

lemmer = nltk.stem.WordNetLemmatizer()

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ('hello', 'hi', 'greetings', 'sup', 'what\'s up', 'hey',)
GREETING_RESPONSES = ['hi', 'hey', 'hi there', 'hello', 'I am glad! You are talking to me']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    sentence_tokens.append(user_response)
    
    vectorizer = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sentence_tokens)
    
    values = cosine_similarity(tfidf[-1], tfidf)
    idx = values.argsort()[0][-2]
    flat = values.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = '{} Sorry, I don\'t understand you'.format(robo_response)
    else:
        robo_response = robo_response + sentence_tokens[idx]
    return robo_response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=['POST'])
def get_response():
    user_response = request.form['user_input']
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            return 'You are welcome...'
        else:
            if (greeting(user_response) != None):
                return greeting(user_response)
            else:
                return response(user_response)
                sentence_tokens.remove(user_response)
    else:
        return 'bye!'

if __name__ == "__main__":
    app.run(debug=True)
