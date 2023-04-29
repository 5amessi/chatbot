import numpy as np
import json
import re
import tensorflow as tf
import random
from flask import Flask
from flask_restx import Resource, Api
import spacy
import python_weather
import asyncio
import os

app = Flask(__name__)
api = Api(app)
nlp = spacy.load("en_core_web_sm")


async def getweather(gpe):
  # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
  async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    # fetch a weather forecast from a city
    weather = await client.get(gpe)
    
    return weather.current.temperature

with open('Intent.json') as f:
    intents = json.load(f)

def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line

# get text and intent title from json data
inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []
        
    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])
        
    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)

def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    
    tokenizer.fit_on_texts(input_list)
    
    input_seq = tokenizer.texts_to_sequences(input_list)

    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')
    
    return tokenizer, input_seq

# preprocess input data
tokenizer, input_tensor = tokenize_data(inputs)

def create_categorical_target(targets):
    word={}
    categorical_target=[]
    counter=0
    for trg in targets:
        if trg not in word:
            word[trg]=counter
            counter+=1
        categorical_target.append(word[trg])
    
    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v,k) for k, v in word.items())

# preprocess output data
target_tensor, trg_index_word = create_categorical_target(targets)

model = tf.keras.models.load_model('chatbot_model')

def response(sentence):
    sent_seq = []
    doc = sentence
    
    # split the input sentences into words
    for token in doc.split():
        if token in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    
    # choice a random response for predicted sentence
    answer =  random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]
    answer = answer[0]
    print(answer)
    print(type(answer))
    doc = nlp(sentence)

    for ent in doc.ents:
        v = "%%"+ent.label_+"%%"
        if v in answer:
            if ent.label_ == "GPE":
                temp = asyncio.run(getweather(ent.text))
                answer = answer.replace("%%TEMP%%" ,str(temp))
            answer = answer.replace(v ,ent.text)
    return answer


@api.route('/chatbot/<string:message>')
class ChatBot(Resource):
    def get(self, message):
        res = response(message)
        print("res", res)
        return str(res)

if __name__ == '__main__':
    app.run()