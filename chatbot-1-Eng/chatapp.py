
## import packages and load pickle files
from os import stat
import nltk
from nltk.stem import WordNetLemmatizer
import pickle 
import numpy as np

import keras
from keras.models import load_model
import json
import random

import tkinter
from tkinter import *

lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

## preprocessing
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    # return bag of words array: 0 or 1 for eacg word in the bag that exists in the sentence
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocab matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocab position
                bag[i] = 1
                if show_details:
                    print('found in bag: %s' % w)
    
    return(np.array(bag))

## predict classes
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

## get appropriate response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    
    return res

## chatbot GUI
def send():
    msg = EntryBox.get('1.0', 'end-1c').strip()
    EntryBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, 'Amel: ' + msg + '\n\n')
        ChatLog.config(foreground='#442265', font=('Verdana', 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, 'Shu-Bot: ' + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title('Chat with Shu-Bot!')
base.geometry('400x500')
base.resizable(width=FALSE, height=FALSE)

# create chat window
ChatLog = Text(base, bd=0, bg='white', height='8', width='50', font='Verdana')

ChatLog.config(state=DISABLED)

# bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor='heart')
ChatLog['yscrollcommand'] = scrollbar.set

# create button to send message
SendButton = Button(base, 
                    font=('Verdana', 12, 'bold'), 
                    text='Send', 
                    height='5', 
                    width='12', 
                    bd=0,
                    bg='grey',
                    #activebackground='#3c9d9b',
                    highlightbackground='#F7CAC9',
                    fg='#F7CAC9', 
                    command=send)

# create box to enter message
EntryBox = Text(base, 
                bd=0,
                bg='white',
                height='5',
                width='29',
                font='Verdana')

# place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()