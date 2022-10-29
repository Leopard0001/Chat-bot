import random
import re
import nltk
import os
import json
import pickle
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import urllib.request
import nest_asyncio





filename = "intents_dataset.json"

# Считываем файл в словарь
with open(filename, 'r', encoding='UTF-8') as file:
    data = json.load(file)

# Создаем массивы фраз и интентов для обучения
X = []
y = []

for name in data:
    for phrase in data[name]['examples']:
        X.append(phrase)
        y.append(name)
    for phrase in data[name]['responses']:
        X.append(phrase)
        y.append(name)

# Создаем объект векторизатора
vectorizer = CountVectorizer()
vectorizer.fit(X)
X_vec = vectorizer.transform(X)


filename2 = "model.pkl"

with open('model.pkl', 'br') as f:
    model = pickle.load(f)



def get_intent(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

def get_response(intent):
    return random.choice(data[intent]['responses'])

def clean_up(text): # очистка текста
    text = text.lower()
    punctuation_re = r"[^\w\s]"
    text = re.sub(punctuation_re, '', text)
    return text

def text_match(user_text, example): # убираем опечатки
    user_text = clean_up(user_text)
    example = clean_up(example)
    if example in user_text:
        return True
    example_len = len(example)
    difference = nltk.edit_distance(user_text, example) / example_len
    return difference < 0.4

def bot(text): # чат бот
    intent = get_intent(text)
    response = get_response(intent)
    return response

intent = None

while intent != 'bye': #запуск чат бота и его основная работа
    text = input('< ')
    print('>', bot(text))
    intent = get_intent(text)




