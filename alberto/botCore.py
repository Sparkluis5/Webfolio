import nltk
import numpy as np
import tensorflow as tf
import random
import json
import pickle
import sys
import Algorithmia
import os
from nltk.stem.lancaster import LancasterStemmer



stemmer = LancasterStemmer()
context = {}
client = Algorithmia.client('XYZ')
algo = client.algo('nlp/SentimentAnalysis/1.0.5')
algo.set_options(timeout=300)

MAX_SEQUENCE_LENGTH = 10
ERROR_THRESHOLD = 0.25



positive_emo = [u"\U0001F600", u"\U0001F602", u"\U0001F603", u"\U0001F604", u"\U0001F606", u"\U0001F607", u"\U0001F609",
             u"\U0001F60A", u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60E", u"\U0001F60F", u"\U0001F31E",
             u"\u263A", u"\U0001F618", u"\U0001F61C", u"\U0001F61D", u"\U0001F61B", u"\U0001F63A", u"\U0001F638",
             u"\U0001F639", u"\U0001F63B", u"\U0001F63C", u"\u2764", u"\U0001F496", u"\U0001F495", u"\U0001F601",
             u"\u2665"]

negative_emo = [u"\U0001F614", u"\U0001F615", u"\u2639", u"\U0001F62B", u"\U0001F629", u"\U0001F622", u"\U0001F625",
              u"\U0001F62A", u"\U0001F613", u"\U0001F62D", u"\U0001F63F", u"\U0001F494"]


def train_model():
    with open('intents.json') as json_data:
        intents = json.load(json_data)

    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words", words)

    with open('addvars.pickle', 'wb') as f:
        pickle.dump([intents, classes, words], f)

    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = np.array(list(training[:,0]))
    train_y = np.array(list(training[:,1]))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(len(train_y[0]), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=1000, batch_size=8)
    model.save('model.trainbot')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


def classify(sentence):
    # generate probabilities from the model
    transformed_sentence = bow(sentence, words)
    results = model.predict(np.array([transformed_sentence]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, userID='123', show_details=False):
    global model, intents, classes, words
    model, intents, classes, words = load_variables()
    EMO_TRESHOLD = 0.30
    results = classify(sentence)
    #print(results)
    #print(context)

    input = {
        "document": sentence
    }

    sentiment = algo.pipe(input).result[0]['sentiment']
    #print(sentiment)

    #emotion = classifyEmotion(sentence.lower())
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        if sentiment > 0.15:
                            if random.uniform(0,1) < EMO_TRESHOLD:
                                return random.choice(i['responses']) + ' ' + random.choice(positive_emo)
                            else:
                                return random.choice(i['responses'])
                        if sentiment < -0.15:
                            if random.uniform(0,1) < EMO_TRESHOLD:
                                return random.choice(i['responses']) + ' ' + random.choice(negative_emo)
                            else:
                                return random.choice(i['responses'])
                        return random.choice(i['responses'])
            results.pop(0)
    else:
        return "NOT FOUND" #Introduzir modulo de geracaoo de pergunta aleatoria

def load_variables():
    model = tf.keras.models.load_model(str(os.getcwd()) + '/alberto/model.trainbot')
    with open(str(os.getcwd()) + '/alberto/addvars.pickle', 'rb') as f:
        intents, classes, words = pickle.load(f)
    return model, intents, classes, words

def load_emotion():
    emo_model = tf.keras.models.load_model('model_lstm_normal.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer, le = pickle.load(f)
    return emo_model, tokenizer, le


if __name__ == "__main__":
    #train_model()
    #model, intents, classes, words = load_variables()
    #emo_model, tokenizer, le = load_emotion()
    while(True):
        sentence = sys.stdin.readline()
        if sentence == 'bye':
            break
        else:
            print(response(sentence))

