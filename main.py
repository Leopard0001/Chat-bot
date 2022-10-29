import random
import re
import nltk

# файл с запросами и ответами
intents = {
    'hello': {
        'examples': ['Хелло', 'Привет', 'Здравствуйте'],
        'responses': ['Добрый день!','Как дела?','Как настроение?']
    },
    'weather': {
        'examples': ['Какая погода?', 'Что за окном?', 'Во что одеваться?'],
        'responses': ['Погода отличная!','У природы нет плохой погоды!'],
    },
    'undefined': {
        'examples': [],
        'responses': ['Извините Я не знаю таких слов(','Давайте поговорим о чем-то другом!'],
    },
    'exit': {
        'examples': ['Выход', 'Пока', 'До свидания'],
        'responses': ['До свидания!','До скорого!','Пока-пока'],
    }
}


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

def get_intent(text): # определить намерение по тексту
    for intent_name in intents: # проверить все существующие intents
        for example in intents[intent_name]['examples']:
            if text_match(text, example):
                return intent_name
    return 'undefined'

def get_response(intent): # случайный response для данного intent
    return random.choice(intents[intent]['responses'])

def bot(text): # чат бот
    intent = get_intent(text)
    response = get_response(intent)
    return response

intent = 'undefined'

while intent != 'exit': #запуск чат бота и его основная работа
    text = input('< ')
    print('>', bot(text))
    intent = get_intent(text)




