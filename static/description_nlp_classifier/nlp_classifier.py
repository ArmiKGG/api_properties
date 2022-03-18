import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import nltk
import pandas as pd


nltk.download('stopword')
nltk.download('punkt')


def make_prediction(model, vectorizer, labels, text):
    index = model.predict(vectorizer.transform(pd.Series(text).apply(process_text)))[0]
    return labels[index]


def process_text(text):
    text = text.lower().replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    text = " ".join(filtered_sentence)
    return text


model_nlp = pickle.load(open('text_classifier.pickle', "rb"))
vectorizer_nlp = pickle.load(open('tfidf.pickle', "rb"))


labels_nlp = ["auction", "good", "rehab"]

description = "auction lot thusday! bid only"
nlp_class = make_prediction(model_nlp, vectorizer_nlp, labels_nlp, description)
print(nlp_class)

import os
import sys
sys.path.append("../../monk_v1/")
sys.path.append("../../monk_v1/monk/")
from monk.gluon_prototype import prototype

gtf = prototype(verbose=1)
gtf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True)


def make_prediction(path):
    img_name = path
    predictions = gtf.Infer(img_name=img_name)
    print(predictions)


make_prediction('../test.jpg')


