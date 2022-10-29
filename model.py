import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import pickle

with open('intents_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# создаем массивы для обучения
X=[]
Y=[]
for intent_name in data:
    for example in data[intent_name]['examples']: #для запросов
        X.append(example)
        Y.append(intent_name)
    for response in data[intent_name]['responses']: #для ответов
        X.append(response)
        Y.append(intent_name)

vectorizer = CountVectorizer()
vectorizer.fit(X)

CountVectorizer()
X_vec = vectorizer.transform(X)


# создаем модель
model = MLPClassifier(max_iter=1000)

# обучаем модель
model.fit(X_vec, Y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
