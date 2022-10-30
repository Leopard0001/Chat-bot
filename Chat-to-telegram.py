import random
import re
import nltk
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nest_asyncio
from telegram.ext import ApplicationBuilder
from telegram.ext import MessageHandler
from telegram.ext import filters

nest_asyncio.apply()
TOKEN = "5568538641:AAHUSFRbz-X28GUqOmhwXmHNPll_lSZHhAU"
filename = "intents_dataset.json"

# Считываем файл в словарь
with open(filename, 'r', encoding='UTF-8') as file:
    data = json.load(file)

# Создаем массивы фраз и интентов для обучения
X = []
y = []

# создаем массив
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

# загружаем модель
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

# вызывать ее при каждом обращении к боту
async def reply(update,context) -> None:
    question = update.message.text
    reply = bot(question)
    print('>', question)
    print('<', reply)
    await update.message.reply_text(reply)

# создаем объект приложения и связываем с токеном
app = ApplicationBuilder().token(TOKEN).build()

#Создаем обработчик текстовых сообщений
handler = MessageHandler(filters.Text(), reply)

# Добавляем обработчик
app.add_handler(handler)

# запускаем приложение: бот крутится, пока крутится колесо выполнения
app.run_polling()





















